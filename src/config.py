import os
from typing import NamedTuple


_PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:

"""

_PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:

"""


VERSION = os.getenv('VERSION', 'v0'),
AUTHOR = os.getenv('AUTHOR', 'kevin')
APP_PORT = os.getenv('APP_PORT', 8000)

# PATH_DATA = os.getenv('PATH_DATA', './temp')
# PATH_MODEL = os.getenv('PATH_MODEL', './model/model.onnx')

TOKENIZER_NAME = os.getnev('TOKENIZER_NAME', 'databricks/dolly-v1-6b')
MODEL_NAME = os.getnev('TOKENIZER_NAME', 'databricks/dolly-v1-6b')
PROMPT = os.getenv('PROMPT', _PROMPT)

GPU = os.getenv('GPU', "-1")

# config = dict(
#     PATH_DATA = PATH_DATA,
#     PATH_MODEL = PATH_MODEL,
#     DELETE_INTERVAL = DELETE_INTERVAL
# )


class Config(NamedTuple):
    app_port: int
    tokenizer_name: str
    model_name: str
    prompt: str
    gpu: str

config_org = Config(
    APP_PORT,
    TOKENIZER_NAME,
    MODEL_NAME,
    PROMPT,
    GPU
)