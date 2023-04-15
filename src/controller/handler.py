import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
import config
import functions


# class Config(NamedTuple):
#     app_port: int
#     tokenizer_name: str
#     model_name: str
#     prompt: str

class model():
    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_name,
            padding_side="left"
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            device_map="auto",
            trust_remote_code=True
        )

    def post_completion(self, **args):
        _prompt = args['prompt']

        functions.generate(
            _prompt,
            
        )

    def post_text(
        self,
        api: str,
        **args
    ) -> dict:
        
        if api == 'post_completion':
            ret = self.post_completion(**args)
        
        else:

        return ret