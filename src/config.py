import os
from typing import NamedTuple


_PROMPT_FMT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:

"""

_PROMPT_FMT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:

"""


VERSION = os.getenv('VERSION', 'v0'),
AUTHOR = os.getenv('AUTHOR', 'kevin')
APP_PORT = os.getenv('APP_PORT', 8000)

# PATH_DATA = os.getenv('PATH_DATA', './temp')
# PATH_MODEL = os.getenv('PATH_MODEL', './model/model.onnx')

# TOKENIZER_NAME = os.getenv('TOKENIZER_NAME', 'databricks/dolly-v1-6b')
# MODEL_NAME = os.getenv('MODEL_NAME', 'databricks/dolly-v1-6b')
# MODEL_NAME = os.getenv('MODEL_NAME', 'geemili/dolly-v2-12b')

TOKENIZER_NAME = os.getenv('TOKENIZER_NAME', 'databricks/dolly-v2-12b')
MODEL_NAME = os.getenv('MODEL_NAME', 'databricks/dolly-v2-12b')


# TOKENIZER_NAME = os.getenv('TOKENIZER_NAME', 'inu-ai/dolly-japanese-gpt-1b')
# MODEL_NAME = os.getenv('MODEL_NAME', 'inu-ai/dolly-japanese-gpt-1b')

PROMPT_FMT = os.getenv('PROMPT', _PROMPT_FMT)

GPU = os.getenv('GPU', "-1")

class Config(NamedTuple):
    app_port: int
    tokenizer_name: str
    model_name: str
    prompt_fmt: str
    gpu: str

config_org = Config(
    APP_PORT,
    TOKENIZER_NAME,
    MODEL_NAME,
    PROMPT_FMT,
    GPU
)