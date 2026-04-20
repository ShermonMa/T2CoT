import config
import sys, os, logging
from datetime import datetime
import shutil
import subprocess
import time
import signal
import asyncio
from concurrent.futures import ThreadPoolExecutor
batch_pool = ThreadPoolExecutor(max_workers=4)

if os.path.exists(config.LOG_FILE):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.move(config.LOG_FILE, f"{config.LOG_BACKUP_PREFIX}{ts}.txt")
project_root = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"🚀 main.py 启动 | 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("-" * 60)


# region 
def start_qwen():  
    import llm.qwen_start
    model, tokenizer = llm.qwen_start.load_model_and_tokenizer()
    sys.modules["__main__"].MODEL = model
    sys.modules["__main__"].TOKENIZER = tokenizer                

    if not os.path.exists(config.QWEN_PID_FILE):
        # 写 PID 文件（供 terminate 用）
        with open(config.QWEN_PID_FILE, "w") as f:
            f.write(str(os.getpid()))
    logger.info("Qwen 模型已在当前进程加载完成.")
    return model, tokenizer

def stop_qwen():
    from llm.qwen_terminate import terminate
    terminate()          # 直接复用 qwen_terminate.py 的逻辑
    logger.info("模型进程已关闭")
# endregion

# region 
def start_gemma():
    import llm.gemma_start  # 触发单例
    model, tokenizer = llm.gemma_start.load_model_and_tokenizer()
    sys.modules["__main__"].MODEL = model
    sys.modules["__main__"].TOKENIZER = tokenizer
    if not os.path.exists(config.GEMMA_PID_FILE):
        with open(config.GEMMA_PID_FILE, "w") as f:
            f.write(str(os.getpid()))
    logger.info("Gemma 模型已在当前进程加载完成.")
    return model, tokenizer
def start_gemma3():
    import llm.gemma3_start  # 触发单例
    model, tokenizer = llm.gemma3_start.load_model_and_tokenizer()
    sys.modules["__main__"].MODEL = model
    sys.modules["__main__"].TOKENIZER = tokenizer

    if not os.path.exists(config.GEMMA3_PID_FILE):
        with open(config.GEMMA3_PID_FILE, "w") as f:
            f.write(str(os.getpid()))
    logger.info("Gemma3 模型已在当前进程加载完成.")
    return model, tokenizer
def start_llama3():
    import llm.llama_start  # 触发单例
    model, tokenizer = llm.llama_start.load_model_and_tokenizer()

    sys.modules["__main__"].MODEL = model
    sys.modules["__main__"].TOKENIZER = tokenizer
    if not os.path.exists(config.LLAMA3_PID_FILE):
        with open(config.LLAMA3_PID_FILE, "w") as f:
            f.write(str(os.getpid()))
    logger.info("LLaMA3 模型已在当前进程加载完成.")
    return model, tokenizer
def start_deepseek():
    import llm.deepseek_start  # 触发单例
    model, tokenizer = llm.deepseek_start.load_model_and_tokenizer()

    sys.modules["__main__"].MODEL = model
    sys.modules["__main__"].TOKENIZER = tokenizer

    if not os.path.exists(config.DEEPSEEK_PID_FILE):
        with open(config.DEEPSEEK_PID_FILE, "w") as f:
            f.write(str(os.getpid()))
    logger.info("DeepSeek 模型已在当前进程加载完成.")
    return model, tokenizer
# endregion


if __name__ == "__main__":
    import argparse, config                    # 把 config 当成模块导进来
    parser = argparse.ArgumentParser()
    parser.add_argument('ds_id', type=int, nargs='?', default=0,
                        choices=config.DATA_MAP.keys())
    args = parser.parse_args()

    config.activate_dataset(args.ds_id)
    from config import QUESTION_FILE, ANSWER_FILE, SCHEMA_FILE, DATASET_NAME

    if(config.QWEN_ENABLE):
        start_qwen()
        from experiment.run_batch_test import run_experiment
        outfile=os.path.join(
        project_root, 
        f"output/{config.DATASET_NAME}", 
        f"qwen_output_raw_{datetime.now():%Y%m%d_%H%M%S}_cot_{config.USE_COT}.jsonl")
        print(f"Output file: {outfile}")
        run_experiment(
            config.QUESTION_FILE,
            outfile=os.path.join(
                project_root, 
                f"output/{config.DATASET_NAME}", 
                f"qwen_output_raw_{datetime.now():%Y%m%d_%H%M%S}_cot_{config.USE_COT}_lora_{config.LoRA_ENABLE}.jsonl"),
            max_new_tokens=512
        )
        stop_qwen()
    if config.GEMMA_ENABLE:
        start_gemma()
        from experiment.run_batch_test import run_experiment
        outfile = os.path.join(
        project_root,
        f"output/{config.DATASET_NAME}",
        f"gemma_output_raw_{datetime.now():%Y%m%d_%H%M%S}_cot_{config.USE_COT}_lora_{config.LoRA_ENABLE}.jsonl"
        ) 
        run_experiment(
            config.QUESTION_FILE,
            outfile,
            max_new_tokens=512
        )
        print("Gemma 模型已关闭")
    if config.GEMMA3_ENABLE:
        start_gemma3()
        from experiment.run_batch_test import run_experiment
        outfile = os.path.join(
        project_root,
        f"output/{config.DATASET_NAME}",
        f"gemma3_output_raw_{datetime.now():%Y%m%d_%H%M%S}_cot_{config.USE_COT}_lora_{config.LoRA_ENABLE}.jsonl"
        ) #
        print(f"Output file: {outfile}")
        run_experiment(
            config.QUESTION_FILE,
            outfile,
            max_new_tokens=512
        )
        print("Gemma 模型已关闭")
    if config.LLAMA3_ENABLE:
        start_llama3()
        from experiment.run_batch_test import run_experiment
        outfile = os.path.join(
        project_root,
        f"output/{config.DATASET_NAME}",
        f"llama3_output_raw_{datetime.now():%Y%m%d_%H%M%S}_cot_{config.USE_COT}_lora_{config.LoRA_ENABLE}.jsonl"
        ) 
        print(f"Output file: {outfile}")
        run_experiment(
            config.QUESTION_FILE,
            outfile,
            max_new_tokens=512
        )
        print("Llama 模型已关闭")
    if config.DEEPSEEK_ENABLE:
        start_deepseek()
        logger.info("模型已加载至主进程，开始批量实验...")
        from experiment.run_batch_test import run_experiment
        outfile = os.path.join(
        project_root,
        f"output/{config.DATASET_NAME}",
        f"deepseek_output_raw_{datetime.now():%Y%m%d_%H%M%S}_cot_{config.USE_COT}_lora_{config.LoRA_ENABLE}.jsonl"
        ) # 区分输出文件
        print(f"Output file: {outfile}")
        run_experiment(
            config.QUESTION_FILE,
            outfile,
            max_new_tokens=512
        )
        print("DeepSeek 模型已关闭")
    