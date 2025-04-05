from flask import Flask, request, jsonify
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from nltk.corpus import stopwords

# Download resource NLTK (agar tidak error saat deploy)
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

app = Flask(__name__)

# Load intents dari file JSON
with open("intents.json", "r") as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

# Siapkan pattern dan respon
patterns = []
responses = []
tags = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        responses.append(intent["responses"])
        tags.append(intent["tag"])

stop_words = set(stopwords.words("indonesian"))

vectorizer = TfidfVectorizer(
    tokenizer=lambda text: [
        lemmatizer.lemmatize(word.lower())
        for word in text.split()
        if word.lower() not in stop_words
    ]
)
pattern_vectors = vectorizer.fit_transform(patterns)


@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    user_input = data.get("question", "")
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, pattern_vectors)
    best_match_idx = np.argmax(similarities)
    best_tag = tags[best_match_idx]
    best_score = similarities[0, best_match_idx]

    if best_score > 0.8:
        matched_intent = next(
            intent for intent in intents["intents"] if intent["tag"] == best_tag
        )
        response = random.choice(matched_intent["responses"])
    else:
        response = "Maaf, saya tidak mengerti pertanyaan Anda."
        with open("unrecognized_questions.txt", "a", encoding="utf-8") as file:
            file.write(user_input + "\n")

    return jsonify({"response": response, "confidence": round(best_score, 3)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
