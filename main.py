from fastapi import FastAPI

app = FastAPI()

@app.get("/embedding/{text}")
def encode_text(text: str):
    return {"text": text}