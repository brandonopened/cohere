from flask import Flask, request, render_template, jsonify
from PyPDF2 import PdfReader
import cohere
import numpy as np

# Initialize Cohere client
co = cohere.ClientV2(api_key="VvImZNZXUXLfUq23reRIyTTlqI2SvRiQOBOwKK18")

app = Flask(__name__)

def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    documents = []
    
    for page in reader.pages:
        try:
            page_text = page.extract_text()
            # Clean up the text
            page_text = ' '.join(line.strip() for line in page_text.split('\n') if line.strip())
            
            # Split into meaningful chunks and filter
            chunks = [chunk.strip() for chunk in page_text.split('.') if chunk.strip()]
            for chunk in chunks:
                if len(chunk) > 50:  # Only keep substantial chunks
                    documents.append({
                        "data": {
                            "text": chunk + "."  # Add back the period
                        }
                    })
                    
        except Exception as e:
            print(f"Error extracting text from page: {e}")
            continue
    
    return documents

def batch_embed_documents(documents, batch_size=96):
    """Embed documents in batches to stay within API limits."""
    all_embeddings = []
    texts = [doc['data']['text'] for doc in documents]
    
    # Process documents in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = co.embed(
            model="embed-english-v3.0",
            input_type="search_document",
            texts=batch,
            embedding_types=["float"]
        ).embeddings.float
        all_embeddings.extend(batch_embeddings)
    
    return np.array(all_embeddings)

# Load and process the PDF
documents = extract_pdf_text("indianadoe.pdf")
print(f"Total documents to embed: {len(documents)}")

# Create embeddings for the documents in batches
doc_embeddings = batch_embed_documents(documents)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    # 1. Embed the query
    query_embedding = co.embed(
        model="embed-english-v3.0",
        input_type="search_query",
        texts=[user_query],
        embedding_types=["float"]
    ).embeddings.float
    
    # 2. Find relevant documents using embeddings
    scores = np.dot(query_embedding, np.transpose(doc_embeddings))[0]
    top_5_idx = np.argsort(-scores)[:5]
    retrieved_documents = [documents[idx] for idx in top_5_idx]
    
    # 3. Rerank the documents
    rerank_results = co.rerank(
        query=user_query,
        documents=[doc['data']['text'] for doc in retrieved_documents],
        top_n=2,
        model='rerank-english-v3.0'
    )
    
    # Get the reranked documents
    reranked_documents = [retrieved_documents[result.index] for result in rerank_results.results]
    
    # 4. Generate the response using Chat
    chat_response = co.chat(
        model="command-r-plus-08-2024",
        messages=[{'role': 'user', 'content': user_query}],
        documents=reranked_documents
    )
    
    # Format the response with citations
    response_text = chat_response.message.content[0].text
    citations = []
    if chat_response.message.citations:
        for citation in chat_response.message.citations:
            citations.append({
                "text": citation.text,
                "source_text": citation.sources[0].document["text"]
            })
    
    return jsonify({
        "response": response_text,
        "citations": citations
    })

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
