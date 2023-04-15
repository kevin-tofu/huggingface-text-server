
from typing import List, Optional
from fastapi import FastAPI, Path, Query

def params_completion(
    prompt: str = '',
    test: Optional[int] = None
):
    """
    """

    ret = {
        'prompt': prompt,
        'test': test,
    }
    return ret


def params_prompt_format(
    prompt_format: str = '',
    test: Optional[int] = None
):
    """
    """

    ret = {
        'prompt_format': prompt_format,
        'test': test,
    }
    return ret
