import argparse
import json
import sys,os
import time
from pathlib import Path
import logging
from typing import List, Dict, Any
# ----------- config 复用 -----------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import config
QWEN_MODEL_PATH = config.QWEN_MODEL_PATH
QWEN_LORA_PATH  = config.QWEN_LORA_PATH
LOG_FILE        = config.QWEN_LOG_FILE

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'),
              logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

import __main__
if not (hasattr(__main__, "MODEL") and hasattr(__main__, "TOKENIZER")):
    logger.warning("❌ 主进程未注册 MODEL / TOKENIZER")
    sys.exit(1)

model = __main__.MODEL
tokenizer = __main__.TOKENIZER
import torch

from transformers import StoppingCriteria, StoppingCriteriaList

class EndOfTurnStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_string: str, tokenizer):
        self.stop_string = stop_string
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        return self.stop_string in decoded
def generate_sync(prompt: str,
                  max_new_tokens: int = config.MAX_NEW_TOKENS,
                  temperature: float = config.TEMPERATURE,
                  top_p: float = config.TOP_P,
                  mode: str = "qwen") -> str:
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(model.device)

    # 1. 选停止符
    if config.GEMMA_ENABLE or config.GEMMA3_ENABLE:
        eos_token_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    elif config.QWEN_ENABLE:
        eos_token_id = tokenizer.eos_token_id
    elif config.LLAMA3_ENABLE or config.DEEPSEEK_ENABLE:
        eos_token_id = tokenizer.eos_token_id
    else: 
        logger.error("generate_sync 未知模式，请检查配置")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=config.TOP_K,
            repetition_penalty=config.REPETITION_PENALTY,
            do_sample=True,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer_ids = out[0, inputs['input_ids'].shape[-1]:]
    return tokenizer.decode(answer_ids, skip_special_tokens=True)
def load_questions(path: Path):
    with path.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def append_result(record: Dict[str, Any], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:      # 'a' 追加
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def run_experiment(infile: str,
                   outfile: str,
                   max_new_tokens: int = config.MAX_NEW_TOKENS,
                   temperature: float = config.TEMPERATURE,
                   top_p: float = config.TOP_P):
    import torch
    import data_process.schema_linking
    import data_process.run_eval
    from pathlib import Path
    if getattr(config, 'QUESTION_WITH_SCHEMA', False):
        q_path = Path(config.QUESTION_FILE)
        s_path = Path(config.QUESTION_WITH_SCHEMA_FILE)
        q_list = [ln.rstrip('\n') for ln in q_path.read_text(encoding='utf8').splitlines()]
        s_list = [json.loads(line)['schema'] for line in s_path.read_text(encoding='utf8').splitlines()]
        assert len(q_list) == len(s_list), 'question.txt 与 schema.jsonl 行数不一致！'
    else:
        q_list = s_list = None
    questions = load_questions(Path(infile))
    out_path = Path(outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")
    if config.QWEN_ENABLE:
        marker = "Please generate a Cypher query based on these fields with no explanations.\n    assistant"
    else:
        marker = "Please generate a Cypher query based on these fields with no explanations.\n"
    question_cnt = 0
    for idx, q in enumerate(questions, 1):
        question_cnt += 1
        if config.TEST_LIMIT != -1 and question_cnt > config.TEST_LIMIT:
            break
        print(f"[{idx}/{len(questions)}] 推理：{q[:50]}...")
        logger.info(f"[{idx}/{len(questions)}] 推理：{q[:50]}...")
        if q_list is not None:                      
            try:
                line_idx = q_list.index(q)          
            except ValueError:
                raise RuntimeError(f'当前问题在 question.txt 中找不到：{q}')
            schema_line = s_list[line_idx]          
            prompt = data_process.schema_linking.run_schema_linking(q, schema_line)
        else:
            prompt = data_process.schema_linking.run_schema_linking(q)

        ans = generate_sync(prompt, max_new_tokens, temperature, top_p)
        print(f"ANSWER: {ans}")
        if marker in ans:
            only_ans = ans.split(marker)[-1].lstrip()
        else:
            logger.warning("输出的答案没有marker")
            only_ans = ans

        append_result({"question": q, "answer": only_ans}, out_path)

    logger.info(f"✅ 结果已写入 {out_path}")
    data_process.run_eval.start()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()
    run_experiment(args.infile, args.outfile,
                   args.max_new_tokens, args.temperature, args.top_p)