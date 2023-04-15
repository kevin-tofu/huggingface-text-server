import os, sys
# print(sys.path)
from fastapi import APIRouter, Depends
from fastapi import BackgroundTasks
# from typing import List, Optional
from config import config_org
from src.router.route_depends import params_completion, params_prompt_format
from controller import handler

myhandler = handler.model(config_org)

router = APIRouter(prefix="")

@router.post('/completion')
def completion(
    params: dict = Depends(params_completion)
):
    """
    Post an image(.jpg ) to make it artistic-style. 
    You can get the artistic-style image using GET /image API. 
    """
    
    # return await handler.post_file("transfer-image", file, "jpg", bgtask, **params)
    return myhandler.post_text("completion", **params)


@router.get('/prompt')
def get_prompt(
    params: dict = Depends(params_completion)
):
    """
    Post an image(.jpg ) to make it artistic-style. 
    You can get the artistic-style image using GET /image API. 
    """
    
    # return await handler.post_file("transfer-image", file, "jpg", bgtask, **params)
    return myhandler.get_text("prompt_fmt", **params)


@router.post('/prompt')
def post_prompt(
    params: dict = Depends(params_completion)
):
    """
    Post an image(.jpg ) to make it artistic-style. 
    You can get the artistic-style image using GET /image API. 
    """
    
    # return await handler.post_file("transfer-image", file, "jpg", bgtask, **params)
    return myhandler.post_text("prompt_fmt", **params)

