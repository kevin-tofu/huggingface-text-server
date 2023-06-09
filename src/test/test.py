
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
import sys
sys.path.append('src')

from controller import functions
from config import config_org


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(
    config_org.tokenizer_name,
    padding_side="left"
)

# `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or 'sequential'
model = AutoModelForCausalLM.from_pretrained(
    config_org.model_name,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)


# tokenizer = AutoTokenizer.from_pretrained(
#     config_org.tokenizer_name,
#     use_fast=False
# )
# model = AutoModelForCausalLM.from_pretrained(
#     config_org.model_name
# ).to(device)



# PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

# ### Instruction:
# {instruction}

# ### Response:
# """

PROMPT_FORMAT = r'<s>\n以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n[SEP]\n指示:\n{instruction}\n[SEP]\n入力:\n{input}\n[SEP]\n応答:\n'
# INSTRUCTION = 'あなたは文脈をもとに'
INSTRUCTION = "あなたは何でも正確に答えられるAIです。"
# INPUT = '将棋界最強の男は誰？.'
INPUT = '日本で一番高い山は？'

prompt = PROMPT_FORMAT.format(instruction=INSTRUCTION, input=INPUT), 
print('prompt:', prompt)

# Sample similar to: "Excited to announce the release of Dolly, a powerful new language model from Databricks! #AI #Databricks"
completion = functions.generate_response(
    prompt,
    model=model,
    tokenizer=tokenizer
)

print('completion:', completion)