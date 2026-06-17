from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

app = FastAPI()
model_name = "multi-qa-mpnet-base-cos-v1"
model = SentenceTransformer(model_name)


@app.get("/embedding")
def encode_text(text: str):
    embedding_binary = model.encode(text, precision="binary")
    return {"embedding_binary": embedding_binary.tolist()}
