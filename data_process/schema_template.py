
import config
from data_process.schema_linking import (     
    load_schema_structured,
    control,
    REL_CYPHER_DICT,
)
QWEN_SYSTEM = (
    "You are CypherAgent, an AI that only translates natural-language questions into Neo4j Cypher queries.\n"
    )
QWEN_SYSTEM_COT =(
    "Let's think step by step.\n"
    "- Identify the nodes, node properties, and relationships in the question.\n"
    "- Match the extracted nodes and their properties to the corresponding schema elements.\n"
    "- Match the extracted relationships to the corresponding schema elements.\n"
    "- If a required field is missing, infer the most likely property name or relationship type based on common naming conventions and context.\n"
)
QWEN_NO_WORDS =(
    "- Return **only** the query string, no explanations, no markdown, no greetings.\n"
)
def _build_qwen_prompt(question: str, schema_text: str, use_cot: bool = False) -> str:
    user_prompt = (
        f"your text to cypher question: {question}\n"
        f"The following are the available fields in the database:\n{schema_text}\n"
        "Please generate a Cypher query based on these fields with no explanations."
    )
    system_prompt = QWEN_SYSTEM
    if use_cot:
        system_prompt += QWEN_SYSTEM_COT
    system_prompt += QWEN_NO_WORDS
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant"
    )

GEMMA_SYSTEM_IN_USER = (
    "You are CypherAgent, an AI that only translates natural-language questions into Neo4j Cypher queries.\n"
)
GEMMA_COT_PROMPT = (
    "Let's think step by step.\n"
    "- Identify the nodes, node properties, and relationships in the question.\n"
    "- Match the extracted nodes and their properties to the corresponding schema elements.\n"
    "- Match the extracted relationships to the corresponding schema elements.\n"
    "- If a required field is missing, infer the most likely property name or relationship type based on common naming conventions and context.\n"
)
GEMMA_NO_WORDS =(
    "- Return **only** the query string, no explanations, no markdown, no greetings.\n"
)
def _build_gemma_prompt(question: str, schema_text: str, use_cot: bool = False) -> str:
    """仅内部使用：把 user-prompt 组装成 Gemma 格式"""
    user_prompt = (
        f"your text to cypher question: {question}\n"
        f"The following are the available fields in the database:\n{schema_text}\n"
        "Please generate a Cypher query based on these fields with no explanations."
    )
    if use_cot:
        user_part = GEMMA_SYSTEM_IN_USER + GEMMA_COT_PROMPT + GEMMA_NO_WORDS + user_prompt
    else:
        user_part = GEMMA_SYSTEM_IN_USER + GEMMA_NO_WORDS+user_prompt
    return (
        "<bos>"
        f"<start_of_turn>user\n{user_part}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
LLAMA_SYSTEM = (
    "You are CypherAgent, an AI that only translates natural-language questions into Neo4j Cypher queries."
)
LLAMA_COT = (
    "Let's think step by step.\n"
    "- Identify the nodes, node properties, and relationships in the question.\n"
    "- Match the extracted nodes and their properties to the corresponding schema elements.\n"
    "- Match the extracted relationships to the corresponding schema elements.\n"
    "- If a required field is missing, infer the most likely property name or relationship type based on common naming conventions and context."
)
LLAMA_NO_WORDS = (
    "Return **only** the query string, no explanations, no markdown, no greetings."
)


def _build_llama_prompt(
    question: str,
    schema_text: str,
    use_cot: bool = False  
) -> str:
    system_content = LLAMA_SYSTEM
    if use_cot:
        system_content += "\n" + LLAMA_COT
    system_content += "\n" + LLAMA_NO_WORDS

    # 2. 构造 user
    user_content = (
        f"your text to cypher question: {question}\n"
        f"The following are the available fields in the database:\n{schema_text}\n"
        "Please generate a Cypher query based on these fields."
    )

    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_content}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
def _build_deepseek_prompt(
    question: str,
    schema_text: str,
    use_cot: bool = False  
) -> str:
    system_content = LLAMA_SYSTEM
    if use_cot:
        system_content += "\n" + LLAMA_COT
    system_content += "\n" + LLAMA_NO_WORDS

    user_content = (
        f"your text to cypher question: {question}\n"
        f"The following are the available fields in the database:\n{schema_text}\n"
        "Please generate a Cypher query based on these fields."
    )

    return f"System: {system_content}\n\nUser: {user_content}\n\nAssistant: "

def build_prompt(question: str,
                 schema: str = "",
                 is_schema_solid: bool = False,
                 model_choice: str = "qwen",
                 use_cot: bool = False) -> str:
    if model_choice.lower() == "qwen":
        prompt = _build_qwen_prompt(question, schema, use_cot=use_cot)
    elif model_choice.lower() == "gemma":
        prompt = _build_gemma_prompt(question, schema, use_cot=use_cot)
    elif model_choice.lower() == "llama":
        prompt = _build_llama_prompt(question, schema, use_cot=use_cot)
    elif model_choice.lower() == "deepseek":
        prompt = _build_deepseek_prompt(question, schema, use_cot=use_cot)
    else:
        raise ValueError(f"Unsupported model_choice: {model_choice}")
    return prompt