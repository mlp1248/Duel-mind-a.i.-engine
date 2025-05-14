"""
Duel Mind Engine Web App with Neural Networks and Persistent Memory
Run with: python duel_mind_web_app.py
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
sentiment_analyzer = pipeline("sentiment-analysis")

MEMORY_DIR = "mind_memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Mind:
    def __init__(self, name):
        self.name = name
        self.model = SimpleNN()
        self.bias = 1.0
        self.memory_file = os.path.join(MEMORY_DIR, f"{name}.json")
        self.memory = self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                return json.load(f)
        return {}

    def save_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=2)

    def extract_features(self, text):
        ascii_values = [ord(c) for c in text[:128]]
        padded = ascii_values + [0] * (128 - len(ascii_values))
        return torch.tensor(padded, dtype=torch.float32)

    def evaluate(self, text):
        if text in self.memory:
            memory_data = self.memory[text]
            return memory_data["decision"], memory_data["score"]

        sentiment = sentiment_analyzer(text)[0]
        sentiment_score = 1 if sentiment["label"] == "POSITIVE" else -1

        x = self.extract_features(text)
        with torch.no_grad():
            output = self.model(x)
        confidence = torch.softmax(output, dim=0)
        decision = "YES" if confidence[0] > confidence[1] else "NO"

        score = float(confidence.max()) * self.bias * sentiment_score
        self.memory[text] = {"decision": decision, "score": score}
        self.save_memory()
        return decision, score

def negotiate(mind1, mind2, text):
    decision1, score1 = mind1.evaluate(text)
    decision2, score2 = mind2.evaluate(text)

    if decision1 == decision2:
        final_decision = decision1
    else:
        final_decision = decision1 if score1 > score2 else decision2

    return {
        "Input": text,
        "Mind1": {"Name": mind1.name, "Decision": decision1, "Score": score1, "Bias": mind1.bias},
        "Mind2": {"Name": mind2.name, "Decision": decision2, "Score": score2, "Bias": mind2.bias},
        "Final Decision": final_decision
    }

@app.route("/duel", methods=["POST"])
def duel():
    data = request.json
    prompt = data.get("prompt", "")
    mind1 = Mind("AlphaNet")
    mind2 = Mind("BetaNet")
    result = negotiate(mind1, mind2, prompt)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
