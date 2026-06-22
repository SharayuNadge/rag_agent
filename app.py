import os
from flask import Flask, render_template, request, jsonify
from query import query_rag, load_index, search, build_context, ask_llm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
index, chunks = load_index()
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    results = search(question, index, chunks)
    pages = list(set([r["page"] for r in results]))
    context = build_context(results)
    answer = ask_llm(question, context)

    return jsonify({
        "answer": answer,
        "pages": pages
    })

if __name__ == "__main__":
    app.run(debug=True, port=8080)