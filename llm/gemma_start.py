
import logging
import os
import sys
from pathlib import Path
from functools import lru_cache
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import config

GEMMA_MODEL_PATH = config.GEMMA_MODEL_PATH
GEMMA_LORA_PATH  = config.GEMMA_LORA_PATH
LOG_FILE         = config.GEMMA_LOG_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        GEMMA_MODEL_PATH, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,   # 计算精度
        device_map={"": "cpu"}        # 关键：全部层放 CPU
    )

    if Path(GEMMA_LORA_PATH).exists() and getattr(config, "LoRA_ENABLE", False):
        logger.info(f"Mounting LoRA on CPU: {GEMMA_LORA_PATH}")
        model = PeftModel.from_pretrained(model, GEMMA_LORA_PATH, device_map={"": "cpu"})
        model = model.merge_and_unload()          # 合并后仍是 cpu 上的原生模型
        logger.info("LoRA merged on CPU.")
    else:
        logger.warning("LoRA path not found or disabled, skipped.")

    logger.info("Moving merged model to GPU …")
    model = model.to("cuda")
    model.eval()

    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    logger.info("Gemma model loaded & switched to eval mode.")
    return model, tokenizer

@lru_cache(maxsize=None)
def _load_once():
    if not Path(GEMMA_MODEL_PATH).exists():
        raise FileNotFoundError(f"Model path not found: {GEMMA_MODEL_PATH}")
    return load_model_and_tokenizer()
tokenizer, model = _load_once()      
import asyncio
from concurrent.futures import ThreadPoolExecutor
_pool = ThreadPoolExecutor(max_workers=1) 
async def generate(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    do_sample: bool = True,
) -> str:
    loop = asyncio.get_event_loop()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = await loop.run_in_executor(
        _pool,
        lambda: model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        ),
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)
