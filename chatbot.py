from flask import Flask, request, jsonify
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity # Mengukur kemiripan antara input pengguna dan dataset intent
import nltk
from nltk.stem import WordNetLemmatizer # Mengubah kata ke bentuk dasar
import numpy as np
from nltk.corpus import stopwords
# nltk.download('stopwords')

# Chatbot ini berbasis Rule-Based dengan menggunakan Flask sebagai back end nya
# Chatbot ini menggunakan TF-IDF (Term Frequency-Inverse Document Frequency) dan Cosine Similarity untuk mencocokkan input pengguna dengan pola yang ada dalam dataset intent

app = Flask(__name__)

# Load intents from dari file JSON, kumpulan intent dari file JSON
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Inisialisasi lemmatizer
lemmatizer = WordNetLemmatizer()

# Siapakan patern dan respon
patterns = [] # Menimpan data pertanyaan
responses = [] # Menyimpan data respon / jawaban
tags = [] # Menimpan kategori dari setiap pola

for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(intent['responses'])
        tags.append(intent['tag'])

# print('patterns',patterns)
# print('responses',responses)
# print('tags',tags)

# Digunakan untuk memotong kalimat yang tidak penting, mengurangi pengaruh kata umum seperti (apa, itu, bagaimana)
stop_words = set(stopwords.words('indonesian'))

# Vectorizer untuk mentransformasi teks ke vector bilangan
vectorizer = TfidfVectorizer(tokenizer=lambda text: [
    lemmatizer.lemmatize(word.lower()) 
    for word in nltk.word_tokenize(text) 
    if word.lower() not in stop_words  # Hilangkan stopwords
])
pattern_vectors = vectorizer.fit_transform(patterns)

# Endpoint API Chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_input = data.get("question", "")

    # Trasformasin inpput dari user ke dalam bentuk vector
    user_vector = vectorizer.transform([user_input])
    
    # Hitung kesamaan kosinus antara input dari user dan dari pola pertanyan yang ada di dalam Intent
    similarities = cosine_similarity(user_vector, pattern_vectors)
    
    # Temukan indeks pola yang paling mirip
    best_match_idx = np.argmax(similarities)
    
    # Mendapatkan tag dan skor kesamaan yang sesuai
    best_tag = tags[best_match_idx]
    best_score = similarities[0, best_match_idx]
    
    # Jika kesamaan berada di atas ambang batas tertentu, kembalikan respons acak yang cocok
    if best_score > 0.8:
        matched_intent = next(intent for intent in intents['intents'] if intent['tag'] == best_tag)
        response = random.choice(matched_intent['responses'])
    else:
        response = "Maaf, saya tidak mengerti pertanyaan Anda."

        # Simpan pertanyaan yang tidak dikenali
        with open("unrecognized_questions.txt", "a", encoding="utf-8") as file:
            file.write(user_input + "\n")

    return jsonify({
        "response": response,
        "confidence" : round(best_score, 3)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
