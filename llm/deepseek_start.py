
import logging
import os,sys
from pathlib import Path
from functools import lru_cache
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import config
DEEPSEEK_MODEL_PATH = config.DEEPSEEK_MODEL_PATH
DEEPSEEK_LORA_PATH  = config.DEEPSEEK_LORA_PATH
LOG_FILE        = config.DEEPSEEK_LOG_FILE

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'),
              logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        config.LLAMA3_MODEL_PATH, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.LLAMA3_MODEL_PATH,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    if Path(config.LLAMA3_LORA_PATH).exists() and config.LoRA_ENABLE is True:
        logger.info(f"挂载 LoRA: {config.LLAMA3_LORA_PATH}")
        model = PeftModel.from_pretrained(model, config.LLAMA3_LORA_PATH)
        model = model.merge_and_unload()         
        logger.info("LoRA 已合并，后续推理无 adapter 开销。")
    else:
        logger.warning("LoRA 路径不存在或未启用，已跳过.")

    model.eval()
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        temperature=0.7,      # 默认温度
        do_sample=True,       # 必须采样
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    logger.info("模型加载完成，进入 eval 模式。")
    return model, tokenizer

if __name__ == "__main__":
    with open("/tmp/deepseek_server.pid", "w") as f:
        f.write(str(os.getpid()))
    logger.info(f"模型进程已启动，PID：{os.getpid()}")
