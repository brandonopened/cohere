from flask import Flask, request, render_template, jsonify
import cohere
import numpy as np
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Cohere client with environment variable
co = cohere.ClientV2(api_key=os.getenv('COHERE_API_KEY'))

# Load and process documents at startup
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    documents = []
    
    for page in reader.pages:
        try:
            page_text = page.extract_text()
            page_text = ' '.join(line.strip() for line in page_text.split('\n') if line.strip())
            chunks = [chunk.strip() for chunk in page_text.split('.') if chunk.strip()]
            for chunk in chunks:
                if len(chunk) > 50:
                    documents.append({
                        "data": {
                            "text": chunk + "."
                        }
                    })
        except Exception as e:
            print(f"Error extracting text from page: {e}")
            continue
    
    return documents

def batch_embed_documents(documents, batch_size=96):
    all_embeddings = []
    texts = [doc['data']['text'] for doc in documents]
    
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

# Load documents and create embeddings at startup
print("Loading and processing PDF...")
documents = extract_pdf_text("indianadoe.pdf")
print(f"Total documents extracted: {len(documents)}")

print("Creating embeddings...")
doc_embeddings = batch_embed_documents(documents)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    query = request.json.get('query', '').strip()
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        # Embed the query
        query_embedding = co.embed(
            model="embed-english-v3.0",
            input_type="search_query",
            texts=[query],
            embedding_types=["float"]
        ).embeddings.float
        
        # Find relevant documents
        scores = np.dot(query_embedding, np.transpose(doc_embeddings))[0]
        top_5_idx = np.argsort(-scores)[:5]
        retrieved_documents = [documents[idx] for idx in top_5_idx]
        
        # Rerank
        rerank_results = co.rerank(
            query=query,
            documents=[doc['data']['text'] for doc in retrieved_documents],
            top_n=2,
            model='rerank-english-v3.0'
        )
        
        # Get reranked documents
        reranked_documents = [retrieved_documents[result.index] for result in rerank_results.results]
        
        # Generate response
        chat_response = co.chat(
            model="command-r-plus-08-2024",
            messages=[{'role': 'user', 'content': query}],
            documents=reranked_documents
        )
        
        # Format response with citations
        citations = []
        if chat_response.message.citations:
            for citation in chat_response.message.citations:
                citations.append({
                    'text': citation.text,
                    'source': citation.sources[0].document['text'][:200] + "..."
                })
        
        return jsonify({
            'answer': chat_response.message.content[0].text,
            'citations': citations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5001)
