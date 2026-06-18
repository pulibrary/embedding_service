from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model_name = "multi-qa-mpnet-base-cos-v1"
model = SentenceTransformer(model_name)


class EmbeddingRequest(BaseModel):
    text: str


@app.post("/embedding")
def encode_text(request: EmbeddingRequest):
    embedding_binary = model.encode(request.text, precision="binary")
    return {"embedding_binary": embedding_binary.tolist()}


# example curl from the console
# curl -X POST http://127.0.0.1:8000/embedding -H "Content-Type: application/json" -d '{"text":"hello world again / I dont want to break it % hi"}'
# {"embedding_binary":[4,38,100,120,-95,-33,114,7,-59,67,-100,49,-10,-82,78,-68,5,-45,113,-71,116,-102,-68,-45,-122,-74,87,94,20,-46,105,66,-119,36,58,78,11,24,47,-56,15,116,-110,-86,-79,63,-107,120,82,-51,71,104,39,-4,-62,-26,-85,-43,60,24,27,65,-108,41,-103,-108,-35,101,91,108,-77,93,113,55,40,-4,-125,6,116,-80,-92,-63,101,-82,-25,-21,63,53,22,102,-57,103,55,9,69,122]}
