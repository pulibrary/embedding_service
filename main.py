from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

app = FastAPI()
model_name = "multi-qa-mpnet-base-cos-v1"
model = SentenceTransformer(model_name)


@app.get("/embedding/{text}")
def encode_text(text: str):
    embedding = model.encode(text)
    return {"embedding": embedding.tolist()}
