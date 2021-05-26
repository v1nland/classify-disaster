# std
import json

# fastapi
from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel

# w2v
import gensim.models.word2vec as w2v

# project files
# import query_model
# from helper.utils import getModelRoute

app = FastAPI()


# @app.post("/models/{model_name}/similar")
# def classify_tweet(model_name: str, query: Optional[str] = None):
#     # model = w2v.Word2Vec.load(getModelRoute(model_name))

#     # result = query_model.MostSimilar(model, query)
#     result = []

#     return {"response": result}
