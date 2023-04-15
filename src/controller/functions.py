import numpy as np
import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)
from typing import Optional

MAX_ASSISTANT_LENGTH = 100
MAX_INPUT_LENGTH = 1024
INPUT_PROMPT = r'<s>\n以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n[SEP]\n指示:\n{instruction}\n[SEP]\n入力:\n{input}\n[SEP]\n応答:\n'
NO_INPUT_PROMPT = r'<s>\n以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n[SEP]\n指示:\n{instruction}\n[SEP]\n応答:\n'


def prepare_input(instruction, input_text):
    if input_text != "":
        prompt = INPUT_PROMPT.format(instruction=instruction, input=input_text)
    else:
        prompt = NO_INPUT_PROMPT.format(instruction=instruction)
    return prompt


def format_output(output):
    output = output.lstrip("<s>").rstrip("</s>").replace("[SEP]", "").replace("\\n", "\n")
    return output


def kwargs_token_default():
    return dict(
        return_tensors='pt'
    )


def kwargs_model_tokenizer(tokenizer):
    return dict(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bad_words_ids=[[tokenizer.unk_token_id]]
    )


def kwargs_model_tokens(size: int):
    return dict(
        min_length=size,
        max_length=min(MAX_INPUT_LENGTH, size + MAX_ASSISTANT_LENGTH),
        temperature=0.7,
        do_sample=True
    )


def kwargs_model_default():
    return dict(
        temperature=0.7,
        do_sample=True
    )


def generate_response(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    kwargs_token: Optional[dict]=None,
    kwargs_model: Optional[dict]=None
):
    _kwargs_token = kwargs_token_default() if kwargs_token is None else kwargs_token_default() | kwargs_token
    token_ids = tokenizer.encode(
        prompt,
        **_kwargs_token
    ).to(model.device)

    # token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    n = len(token_ids[0])

    with torch.no_grad():
        
        _kwargs_model_default = kwargs_model_default() if kwargs_model is None else kwargs_model
        _kwargs_model = _kwargs_model_default | kwargs_model_tokens(n) | kwargs_model_tokenizer(tokenizer)
        output_ids = model.generate(
            token_ids.to(model.device),
            **_kwargs_model
        )
        
    output = tokenizer.decode(output_ids.tolist()[0])
    # formatted_output_all = format_output(output)
    # response = f"Assistant:{formatted_output_all.split('応答:')[-1].strip()}"

    return output


# ImportError: Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: `pip install accelerate`

def generate_test(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    do_sample: bool = True,
    max_new_tokens: int = 256,
    top_p: float = 0.92,
    top_k: int = 0,
    **kwargs
) -> str:
    
    # input_ids = tokenizer(prompt_format.format(instruction=instruction), return_tensors="pt").input_ids # .to("cuda")
    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    ).input_ids.to(model.device)

    # each of these is encoded to a single token
    response_key_token_id = tokenizer.encode("### Response:")[0]
    end_key_token_id = tokenizer.encode("### End")[0]

    gen_tokens = model.generate(
        input_ids,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=end_key_token_id,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        **kwargs
    )[0].cpu()
    print('gen_tokens:', gen_tokens)
    # find where the response begins
    response_positions = np.where(gen_tokens == response_key_token_id)[0]

    if len(response_positions) >= 0:
        response_pos = response_positions[0]
        
        # find where the response ends
        end_pos = None
        end_positions = np.where(gen_tokens == end_key_token_id)[0]
        if len(end_positions) > 0:
            end_pos = end_positions[0]

        return tokenizer.decode(gen_tokens[response_pos + 1 : end_pos]).strip()

    return None
