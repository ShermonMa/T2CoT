from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_path = "/root/autodl-tmp/llm/qwen/qwen/Qwen2.5-7B-Instruct" 
lora_path = "/root/autodl-tmp/fine-tuning/qwen" 

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model_base = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model_base.eval()

model_lora = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model_lora = PeftModel.from_pretrained(model_lora, lora_path)
model_lora.eval()

test_cases = [
]

def generate_cypher(model, tokenizer, schema, question):
    prompt = f"""用户
根据以下 Neo4j 数据库 Schema：
{schema}

请将这个自然语言问题转换为 Cypher 查询语句：
{question}

助手
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.1,
        top_p=0.9
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "助手" in response:
        cypher = response.split("助手")[-1].strip()
    else:
        cypher = response
    return cypher
for i, case in enumerate(test_cases):
    cypher_base = generate_cypher(model_base, tokenizer, case["schema"], case["question"])
    print(cypher_base)

    cypher_lora = generate_cypher(model_lora, tokenizer, case["schema"], case["question"])
    print(cypher_lora)