
import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import argparse
from pathlib import Path

TRAIN_DATA_PATH = "/root/autodl-tmp/T2CoT/data/text2cypher/train_split_qwen_no_cot.jsonl"
EVAL_DATA_PATH  = "/root/autodl-tmp/T2CoT/data/text2cypher/val_split_qwen_no_cot.jsonl"
MODEL_PATH      = "/root/autodl-tmp/llm/qwen/qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR      = "/root/autodl-tmp/fine-tuning/qwen_no_cot_epoch_1.5"

class EarlyStopOnPlateauCallback(TrainerCallback):
    """
    1. 验证 loss 连续 patience 次不下降 -> 提前停止
    2. 优化内存管理，减少验证时的GPU占用
    """
    def __init__(self, patience: int = 4):          # 默认 4 次
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.best_ckpt_path = None

    def on_evaluate(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):
        # 早停判断
        eval_loss = None
        for log in reversed(state.log_history):
            if "eval_loss" in log:
                eval_loss = log["eval_loss"]
                break
        if eval_loss is None:
            return

        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.counter = 0
            self.best_ckpt_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                control.should_training_stop = True
                print(f"\n*** 验证 loss 连续 {self.patience} 次未下降，停止训练 ***")

def create_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # 移除flash_attention_2配置，使用默认注意力机制
    )
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer

def create_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

def load_dataset_jsonl(path, tokenizer, max_length=2048, filter_max_tokens=1024):
    """单独给 train / eval 分别调用，并可选过滤过长样本"""
    raw = load_dataset("json", data_files=path, split="train")

    def _tokenize_and_add_length(example):
        text = example["question"] + example["answer"]
        tokenized = tokenizer(text, truncation=False, padding=False)  
        input_ids = tokenized["input_ids"]
        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"],
            "labels": input_ids.copy(),
            "length": len(input_ids)  
        }

    ds = raw.map(
        _tokenize_and_add_length,
        remove_columns=raw.column_names,
        desc="Tokenizing and computing lengths"
    )

    ds = ds.filter(
        lambda x: x["length"] <= filter_max_tokens,
        desc=f"Filtering samples longer than {filter_max_tokens} tokens"
    )

    def _maybe_truncate(example):
        if len(example["input_ids"]) > max_length:
            example["input_ids"] = example["input_ids"][:max_length]
            example["attention_mask"] = example["attention_mask"][:max_length]
            example["labels"] = example["labels"][:max_length]
        return example

    if max_length < filter_max_tokens:
        ds = ds.map(_maybe_truncate, desc="Truncating to max_length")

    # ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "length"])
    return ds
class MemoryEfficientTrainer(Trainer):

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        torch.cuda.empty_cache()
        self.model.eval()
        result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        torch.cuda.empty_cache()
        
        return result

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("加载模型 / tokenizer ...")
    model, tokenizer = create_model_and_tokenizer()
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("加载数据集 ...")
    train_dataset = load_dataset_jsonl(TRAIN_DATA_PATH, tokenizer, max_length=2048, filter_max_tokens=1024)  # 减小长度
    eval_dataset  = load_dataset_jsonl(EVAL_DATA_PATH,  tokenizer, max_length=2048, filter_max_tokens=1024)  # 减小长度
    print(f"✅ 过滤后训练集样本数: {len(train_dataset)}")
    print(f"✅ 过滤后验证集样本数: {len(eval_dataset)}")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=1,       
        per_device_eval_batch_size=1,         
        eval_accumulation_steps=16,          
        dataloader_pin_memory=False,          
        gradient_accumulation_steps=16,      
        learning_rate=5e-4,
        num_train_epochs=1.5,
        warmup_steps=100,
        logging_steps=10,
        eval_steps=400,
        eval_strategy="steps",
        save_steps=400,
        save_total_limit=2,                  
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        dataloader_drop_last=True,
        dataloader_num_workers=2,           
        group_by_length=True,
        length_column_name="length",
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
        report_to=None,                       
        remove_unused_columns=False,
        seed=42,
        fp16_full_eval=True,                  
        dataloader_prefetch_factor=1,      
        torch_compile=False,                  
    )

    trainer = MemoryEfficientTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStopOnPlateauCallback(patience=4)],
    )
    
    torch.cuda.empty_cache()
    eval_results = trainer.evaluate()
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    train()

if __name__ == "__main__":
    main()



