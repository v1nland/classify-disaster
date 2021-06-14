# std
import json

# fastapi
from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel

# w2v
import gensim.models.word2vec as w2v

# project files
from classify import Classify

# from helper.utils import getModelRoute

app = FastAPI()


class ClassifyTweetRequest(BaseModel):
    tweet_data: str


@app.post("/models/{model_name}/classify")
def exists(model_name: str, req: ClassifyTweetRequest, dim: Optional[int] = 10):
    tweet_type = Classify(model_name, dim, 0, req.tweet_data)

    return {"type": tweet_type}
