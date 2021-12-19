# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment?text=I+like+you.+I+love+you
import csv
import urllib.request

import numpy as np
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


class PredictSentiment:

    def __init__(self):
        self.text = None
        self.task = "emotion"
        MODEL = f"cardiffnlp/twitter-roberta-base-{self.task}"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        # download label mapping
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{self.task}/mapping.txt"
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode('utf-8').split("\n")
            csvreader = csv.reader(html, delimiter='\t')
        self.labels = [row[1] for row in csvreader if len(row) > 1]
        # PT
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        # model.save_pretrained(MODEL)

    def preprocess_text(self):
        """Preprocess text (username and link placeholders)"""
        new_text = []

        for t in self.text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def run_model(self, text, task=None):
        """ Tasks: emoji, emotion, hate, irony, offensive, sentiment"""
        self.text = text
        if task:
            self.task = task
        text = self.preprocess_text()
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        return self.get_results(scores, self.labels)

    def get_results(self, scores, labels):

        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for i in range(scores.shape[0]):
            l = labels[ranking[i]]
            s = scores[ranking[i]]
            if self.task == "emoji":
                return f"Этот смайлик {self.text} - похож на {l} с вероятностью {np.round(float(s), 4)}"
            else:
                return f"Фраза '{self.text}' - относится к категории {l} с вероятностью {np.round(float(s), 4)}"


# p = Predict_Sentiment().run_model(text="I love you very")
# print(p)
