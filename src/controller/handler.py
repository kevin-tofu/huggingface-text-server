import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # PreTrainedModel,
    # PreTrainedTokenizer
)
import config
import functions
from src.logconf import mylogger
logger = mylogger(__name__)

# class Config(NamedTuple):
#     app_port: int
#     tokenizer_name: str
#     model_name: str
#     prompt: str

class model():
    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self.prompt_fmt = cfg.prompt_fmt

        mylogger.info(f"tokenizer_name : {cfg.tokenizer_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_name,
            padding_side="left"
        )

        mylogger.info(f"model_name : {cfg.model_name}")
        self._model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            device_map="auto",
            trust_remote_code=True
        )


    def get_prompt_fmt(self, **kwargs) -> str:
        ret = self.prompt_fmt.format(instruction='here-is-your-instruction')
        return ret


    def post_prompt_fmt(self, **kwargs) -> str:
        self.prompt_fmt = kwargs['prompt']
        return dict(status='ok')


    def post_completion(self, **kwargs):
        _prompt = kwargs['prompt']

        ret = functions.generate(
            _prompt,
            self._tokenizer,
            self._model
        )


    def get_text(
        self,
        api: str,
        **kwargs
    ) -> dict:
        
        mylogger.info(f"get_text - api : {api}")
        if api == 'prompt_fmt':
            ret = self.get_prompt_fmt(**kwargs)
        else:
            raise ValueError('api error')
        
        return ret


    def post_text(
        self,
        api: str,
        **kwargs
    ) -> dict:
        
        mylogger.info(f"post_text - api : {api}")
        if api == 'completion':
            ret = self.post_completion(**kwargs)
        
        elif api == 'prompt_fmt':
            ret = self.post_prompt_fmt(**kwargs)

        else:
            raise ValueError('api error')
        
        return ret

