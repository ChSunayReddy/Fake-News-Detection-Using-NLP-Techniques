from flask import Flask, render_template, request
import numpy as np
import joblib
import re
import string
import pandas as pd
import os

app = Flask(__name__)

# Correct the path to the model file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
model_path = os.path.join(parent_dir, "Model.pkl")
print(f"Corrected model path: {model_path}")

# Load the model
Model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template("index.html")

def wordpre(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)  # remove special characters
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


@app.route('/', methods=['POST'])
def pre():
    raw_text = request.form['txt']

    # Split multiple articles
    articles = [a.strip() for a in raw_text.split("===") if a.strip()]
    final_results = []

    for article in articles:
        sentences = split_sentences(article)
        sentence_results = []

        probs = []

        for sent in sentences:
            cleaned = wordpre(sent)
            cleaned = pd.Series([cleaned])

            pred = Model.predict(cleaned)[0]

            # Confidence score (IMPORTANT)
            if hasattr(Model, "predict_proba"):
                confidence = max(Model.predict_proba(cleaned)[0])
            else:
                confidence = 1.0  # fallback

            probs.append(confidence)

            sentence_results.append({
                "sentence": sent,
                "prediction": pred,
                "confidence": round(confidence * 100, 2)
            })

        # Article-level decision
        avg_confidence = round(np.mean(probs) * 100, 2)
        true_votes = sum(1 for s in sentence_results if s["prediction"] == 1)
        fake_votes = len(sentence_results) - true_votes

        article_label = "TRUE" if true_votes >= fake_votes else "FAKE"
        article_pred = 1 if article_label == "TRUE" else 0

        final_results.append({
            "article_text": article,
            "article_label": article_label,
            "article_prediction": article_pred,
            "article_confidence": avg_confidence,
            "sentences": sentence_results
        })

    return render_template("index.html", results=final_results)

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# from flask import Flask, render_template, request
# import joblib
# import re
# import string
# import pandas as pd
# import os

# app = Flask(__name__)

# # Correct the path to the model file
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# model_path = os.path.join(parent_dir, "Model.pkl")
# print(f"Corrected model path: {model_path}")

# # Load the model
# Model = joblib.load(model_path)

# @app.route('/')
# def index():
#     return render_template("index.html")

# def wordpre(text):
#     text = text.lower()
#     text = re.sub(r'\[.*?\]', '', text)
#     text = re.sub("\\W", " ", text)  # remove special characters
#     text = re.sub(r'https?://\S+|www\.\S+', '', text)
#     text = re.sub('<.*?>+', '', text)
#     text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub('\n', '', text)
#     text = re.sub(r'\w*\d\w*', '', text)
#     return text

# @app.route('/', methods=['POST'])
# def pre():
#     if request.method == 'POST':
#         txt = request.form['txt']
#         txt = wordpre(txt)
#         txt = pd.Series([txt])  # Ensure this is passed as a list or a Series
        
#         try:
#             result = Model.predict(txt)
#             result = result[0]  # Assuming it's a list/array, get the first prediction
#         except Exception as e:
#             result = f"Error: {str(e)}"
        
#         return render_template("index.html", result=result)

# if __name__ == "__main__":
#     app.run(debug=True)
