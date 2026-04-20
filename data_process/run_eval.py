
import json
import sys,os
from pathlib import Path
import logging
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
eval_root = os.path.dirname(os.path.abspath(__file__))
if eval_root not in sys.path:
    sys.path.insert(0, eval_root)
import eval_exact_match
import eval_BLEU
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

def start():
    answer_path = config.ANSWER_FILE
    output_path = config.OUTPUT_FILE
    EM_path = config.EM_EVAL_FILE
    BLEU_path = config.BLEU_EVAL_FILE
    eval_exact_match.compute_exact_match(answer_path,output_path,EM_path)
    eval_BLEU.eval_bleu(answer_path,output_path,BLEU_path)
def start_qwen():
    answer_path = config.ANSWER_FILE
    output_path = "/root/autodl-tmp/T2CoT/output/output_qwen.jsonl"
    EM_path = "/root/autodl-tmp/T2CoT/output/eval_qianwen_em.txt"
    BLEU_path = "/root/autodl-tmp/T2CoT/output/eval_qianwengpt_bleu.txt"
    eval_exact_match.compute_exact_match(answer_path,output_path,EM_path)
    eval_BLEU.eval_bleu(answer_path,output_path,BLEU_path)
if __name__ == '__main__':
    start()