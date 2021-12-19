from fastapi import FastAPI
from pydantic import BaseModel

from sentiment_analysis import PredictSentiment


class Item(BaseModel):
    text: str
    task: str


app = FastAPI()
pred = PredictSentiment()


@app.post("/predict/")
def predict(item: Item):
    """You can choose the following options(task): emoji, emotion, hate, irony, offensive, sentiment./
    Examples: I am very angry with him, I am crying, It's such a beautiful spring day today, stupid man,
    Thank you, Captain Obvious!, 😊 😬 😂 😉 😘 😤 😜 😂 ❤ 💕"""
    return pred.run_model(item.text, item.task)

# predict("I am very angry with him")
# text = "I am very angry with him"
# text = "I am crying"
# text = "It's such a beautiful spring day today"
# text = "You will be able to pass the exam"
# text = "I am very angry with him"
# text = "stupid man"
# text = "I hate everyone who writes theatre blogs."
# text = "😡"
# text = "😀 😡 😉 😘 😤 😜 😂 ❤"
# uvicorn ml_project:app
