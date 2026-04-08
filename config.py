

QWEN_ENABLE =False
GEMMA_ENABLE = False
GEMMA3_ENABLE = False
LLAMA3_ENABLE = False
DEEPSEEK_ENABLE = True
LoRA_ENABLE = False
USE_COT =False 

#------------------QWEN------------
QWEN_MODEL_PATH = "/root/autodl-tmp/llm/qwen/qwen/Qwen2.5-7B-Instruct"
QWEN_LORA_PATH = "/root/autodl-tmp/fine-tuning/qwen_no_cot_epoch_1.5"
QWEN_LOG_FILE = "/root/autodl-tmp/T2CoT/log/model_log/qwen.log"
QWEN_SERVER_LOG = "/root/autodl-tmp/T2CoT/log/model_log/server.log"
QWEN_PID_FILE = "/root/autodl-tmp/T2CoT/log/qwen.pid"

#------------------GEMMA------------
GEMMA_MODEL_PATH = "/root/autodl-tmp/llm/gemma"
GEMMA_LORA_PATH = "/root/autodl-tmp/fine-tuning/gemma-cot-ver2-epoch-1.5"
GEMMA_LOG_FILE = "/root/autodl-tmp/T2CoT/log/model_log/gemma.log"
GEMMA_SERVER_LOG = "/root/autodl-tmp/T2CoT/log/model_log/server.log"
GEMMA_PID_FILE = "/root/autodl-tmp/T2CoT/log/gemma.pid"

#-----------------GEMMA3---------------
GEMMA3_MODEL_PATH = "/root/autodl-tmp/llm/gemma3"
GEMMA3_LORA_PATH = "/root/autodl-tmp/fine-tuning/gemma-3-4b-cot"
GEMMA3_LOG_FILE = "/root/autodl-tmp/T2CoT/log/model_log/gemma3.log"
GEMMA3_SERVER_LOG = "/root/autodl-tmp/T2CoT/log/model_log/server.log"
GEMMA3_PID_FILE = "/root/autodl-tmp/T2CoT/log/gemma3.pid"

#-----------------LLAMA3---------------
LLAMA3_MODEL_PATH = "/root/autodl-tmp/llm/llama"
LLAMA3_LORA_PATH = "/root/autodl-tmp/fine-tuning/llama-3-8b-nocot"
LLAMA3_LOG_FILE = "/root/autodl-tmp/T2CoT/log/model_log/llama3.log"
LLAMA3_SERVER_LOG = "/root/autodl-tmp/T2CoT/log/model_log/server.log"
LLAMA3_PID_FILE = "/root/autodl-tmp/T2CoT/log/llama3.pid"

#-----------------DEEPSEEK CODER---------------
DEEPSEEK_MODEL_PATH = "/root/autodl-tmp/llm/deepseek-6dot7b"
DEEPSEEK_LORA_PATH = "/root/autodl-tmp/fine-tuning/deepseek-cot"
DEEPSEEK_LOG_FILE = "/root/autodl-tmp/T2CoT/log/model_log/deepseek.log"
DEEPSEEK_SERVER_LOG = "/root/autodl-tmp/T2CoT/log/model_log/server.log"
DEEPSEEK_PID_FILE = "/root/autodl-tmp/T2CoT/log/deepseek.pid"
SBERT_MODEL_PATH = "/root/autodl-tmp/schema_linking/all-MiniLM-L6-v2"
NEO4J_PASSWORD= ""
NEO4J_DATABASE= ""



#------------------LOG FILES------------------------
LOG_PATH = "/root/autodl-tmp/T2CoT/log/"
LOG_FILE = "/root/autodl-tmp/T2CoT/log/log.log"
LOG_BACKUP_PREFIX = "/root/autodl-tmp/T2CoT/log/log_"


#---------------INPUT FILES--------------
DATA_MAP = {
    0: {
        "QUESTION_FILE": "/root/autodl-tmp/T2CoT/data/synthcypher_test/question.txt",
        "ANSWER_FILE":   "/root/autodl-tmp/T2CoT/data/synthcypher_test/answer.txt",
        "SCHEMA_FILE":   "/root/autodl-tmp/T2CoT/data/synthcypher_test/generate_schema.json",
        "DATASET_NAME": "synthcypher"
    },
    1: {
        "QUESTION_FILE": "/root/autodl-tmp/T2CoT/data/synthetic-text2cypher-gpt4turbo/question.txt",
        "ANSWER_FILE":   "/root/autodl-tmp/T2CoT/data/synthetic-text2cypher-gpt4turbo/cypher.txt",
        "SCHEMA_FILE":   "/root/autodl-tmp/T2CoT/data/synthetic-text2cypher-gpt4turbo/new_schema.json",
        "DATASET_NAME": "gpt_turbo"
    },
    2: {
        "QUESTION_FILE": "/root/autodl-tmp/T2CoT/data/text2cypher/all_test_questions_text2cypher.txt",
        "ANSWER_FILE":   "/root/autodl-tmp/T2CoT/data/text2cypher/all_test_answer_text2cypher.txt",
        "SCHEMA_FILE":   "/root/autodl-tmp/T2CoT/data/text2cypher/new_schema.json",
        "DATASET_NAME": "text2cypher"
    },
}
def activate_dataset(ds_id: int):
    if ds_id not in DATA_MAP:
        raise KeyError(f'ds_id {ds_id} 不存在')
    for k, v in DATA_MAP[ds_id].items():
        globals()[k] = v
QUESTION_FILE = ""
ANSWER_FILE   = ""
OUTPUT_DIR = "/root/autodl-tmp/T2CoT/output"
DATASET_NAME = "default_dataset"

import os

SCHEMA_FILE = "/root/autodl-tmp/T2CoT/data/text2cypher/new_schema.json"
QUESTION_WITH_SCHEMA_FILE = "/root/autodl-tmp/T2CoT/data/synthcypher_test/schema.jsonl"
QUESTION_WITH_SCHEMA = False 

#----------------output files-------------------------
OUTPUT_FILE = "/root/autodl-tmp/T2CoT/output/output_raw.jsonl"
EM_EVAL_FILE = "/root/autodl-tmp/T2CoT/output/exact_match.txt"
BLEU_EVAL_FILE = "/root/autodl-tmp/T2CoT/output/bleu.txt"
#-----------------MODEL PATAMETERS-------------------------
SBERT_THRESHOLD = 0.6  
SBERT_TOP_K = 3 
SBERT_MIN_SCORE = 0.6  
MAX_NEW_TOKENS = 200  
TEMPERATURE = 0.1  
TOP_P = 0.9  
TOP_K = 50  
REPETITION_PENALTY = 1.1 
#----------------TEST PATAMETERS--------------------------
TEST_LIMIT = -1  # 限制测试样本数量，-1为不限制


