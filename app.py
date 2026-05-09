import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)
CORS(app)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def load_kb():
    word_df = pd.read_csv("knowledge_base/word_vocab.csv")
    phrase_df = pd.read_csv("knowledge_base/phrase_vocab.csv")
    sentence_df = pd.read_csv("knowledge_base/sentence_vocab.csv")

    kb = pd.concat([word_df, phrase_df, sentence_df], ignore_index=True)
    return kb


knowledge_base = load_kb()


def find_matching_kb_terms(input_text):
    matched_terms = []

    for _, row in knowledge_base.iterrows():
        english = str(row["English"]).strip()
        sinhala = str(row["Sinhala"]).strip()

        if english and english.lower() in input_text.lower():
            matched_terms.append({
                "english": english,
                "sinhala": sinhala
            })

    return matched_terms[:50]


@app.route("/", methods=["GET"])
def home():
    return "Python Sinhala Translator Backend is running"


@app.route("/translate", methods=["POST"])
def translate():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text.strip():
            return jsonify({
                "success": False,
                "error": "Privacy policy text is required"
            }), 400

        matched_terms = find_matching_kb_terms(text)

        kb_context = ""
        for item in matched_terms:
            kb_context += f"- {item['english']} = {item['sinhala']}\n"

        prompt = f"""
You are a Sinhala legal translation assistant.

Translate the following English privacy policy into Sinhala.

Use the provided knowledge base terms strictly when they match the text.

Knowledge Base Terms:
{kb_context}

Translation Requirements:
- Preserve legal meaning
- Use accurate privacy and legal terminology
- Use simple and clear Sinhala
- Do not add extra explanation
- Only output Sinhala translation

Privacy Policy:
{text}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )

        return jsonify({
            "success": True,
            "translation": response.text,
            "matched_terms": matched_terms
        })

    except Exception as e:
        print(e)
        return jsonify({
            "success": False,
            "error": "Translation failed"
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)