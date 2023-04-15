
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from instruct_pipeline import InstructionTextGenerationPipeline

import torch
import sys
sys.path.append('src')

from controller import functions
from config import config_org


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = AutoTokenizer.from_pretrained(
#     config_org.tokenizer_name,
#     padding_side="left"
# )

# # `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or 'sequential'
# model = AutoModelForCausalLM.from_pretrained(
#     config_org.model_name,
#     device_map="auto",
#     trust_remote_code=True
# ).to(device)


generate_text = pipeline(
    model="databricks/dolly-v2-12b", 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True,
    device_map="auto"
)



# tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b", padding_side="left")
# model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b", device_map="auto")

generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)




# tokenizer = AutoTokenizer.from_pretrained(
#     config_org.tokenizer_name,
#     use_fast=False
# )
# model = AutoModelForCausalLM.from_pretrained(
#     config_org.model_name
# ).to(device)



PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

INSTRUCTION = "What is the tallest mountain in the world?"

prompt = PROMPT_FORMAT.format(instruction=INSTRUCTION) 
print('prompt:', prompt)

# Sample similar to: "Excited to announce the release of Dolly, a powerful new language model from Databricks! #AI #Databricks"
completion = functions.generate_test(
    prompt,
    model=model,
    tokenizer=tokenizer
)

print('completion:', completion)