from flask import Flask, request, render_template, jsonify
from PyPDF2 import PdfReader
import cohere

# Initialize Cohere client
client = cohere.Client(api_key="VvImZNZXUXLfUq23reRIyTTlqI2SvRiQOBOwKK18")

app = Flask(__name__)

# Function to extract text from PDF
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to query the Cohere ReRank API
def rerank_query(query, documents):
    response = client.rerank(
        model="rerank-v3.5",
        query=query,
        documents=documents,
        top_n=3
    )
    return response

# Load and process the PDF
pdf_text = extract_pdf_text("indianadoe.pdf")
documents = pdf_text.split("\n")  # Split the text into lines or chunks for reranking

@app.route("/")
def home():
    return render_template("index.html")  # A simple HTML form for querying

@app.route("/query", methods=["POST"])
def query():
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    # Call the Cohere ReRank API
    response = rerank_query(user_query, documents)
    ranked_results = [{"text": doc.text, "score": doc.relevance_score} for doc in response.results]

    return jsonify({"results": ranked_results})

if __name__ == "__main__":
    app.run(debug=True)
