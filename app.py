from flask import Flask, request, render_template, jsonify
import cohere
import numpy as np
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import glob

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Cohere client with environment variable
co = cohere.ClientV2(api_key=os.getenv('COHERE_API_KEY'))

def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    documents = []
    
    print(f"Processing {pdf_path}...")
    for page in reader.pages:
        try:
            page_text = page.extract_text()
            page_text = ' '.join(line.strip() for line in page_text.split('\n') if line.strip())
            chunks = [chunk.strip() for chunk in page_text.split('.') if chunk.strip()]
            for chunk in chunks:
                if len(chunk) > 50:  # Only keep substantial chunks
                    documents.append({
                        "data": {
                            "text": chunk + ".",
                            "source": os.path.basename(pdf_path)  # Add source PDF filename
                        }
                    })
        except Exception as e:
            print(f"Error extracting text from page in {pdf_path}: {e}")
            continue
    
    return documents

def batch_embed_documents(documents, batch_size=96):
    """Embed documents in batches to stay within API limits."""
    all_embeddings = []
    texts = [doc['data']['text'] for doc in documents]
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Embedding batch {i//batch_size + 1} of {len(texts)//batch_size + 1}")
        batch_embeddings = co.embed(
            model="embed-english-v3.0",
            input_type="search_document",
            texts=batch,
            embedding_types=["float"]
        ).embeddings.float
        all_embeddings.extend(batch_embeddings)
    
    return np.array(all_embeddings)

# Load all PDFs and create embeddings at startup
print("Loading and processing PDFs...")
documents = []
pdf_files = glob.glob("*.pdf")
print(f"Found PDFs: {pdf_files}")

for pdf_file in pdf_files:
    print(f"\nProcessing {pdf_file}...")
    try:
        new_docs = extract_pdf_text(pdf_file)
        documents.extend(new_docs)
        print(f"Added {len(new_docs)} chunks from {pdf_file}")
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")

print(f"\nTotal documents extracted: {len(documents)}")

# Add this after PDF processing
print("\nDocument sources breakdown:")
source_counts = {}
for doc in documents:
    source = doc['data'].get('source', 'unknown')
    source_counts[source] = source_counts.get(source, 0) + 1
    
for source, count in source_counts.items():
    print(f"{source}: {count} chunks")

# Add this before embedding
print(f"\nTotal documents to embed: {len(documents)}")
print("First few documents:")
for i, doc in enumerate(documents[:3]):
    print(f"\nDoc {i + 1}:")
    print(f"Source: {doc['data'].get('source', 'unknown')}")
    print(f"Text preview: {doc['data']['text'][:100]}...")

print("\nCreating embeddings...")
doc_embeddings = batch_embed_documents(documents)

@app.route('/')
def home():
    # Pass the list of loaded PDFs to the template
    pdf_files = [pdf for pdf in glob.glob("*.pdf")]
    return render_template('index.html', pdf_files=pdf_files)

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
        
        # Find relevant documents from each source
        all_retrieved_docs = []
        unique_sources = set(doc['data']['source'] for doc in documents)
        print(f"Querying across {len(unique_sources)} documents: {unique_sources}")  # Debug print
        
        for source in unique_sources:
            # Get indices for this source
            source_indices = [i for i, doc in enumerate(documents) if doc['data']['source'] == source]
            source_embeddings = doc_embeddings[source_indices]
            
            # Calculate scores for this source's documents
            scores = np.dot(query_embedding, np.transpose(source_embeddings))[0]
            # Get top 5 from this source
            top_k_idx = np.argsort(-scores)[:5]
            # Convert back to original document indices
            top_docs_indices = [source_indices[idx] for idx in top_k_idx]
            # Add top docs from this source
            all_retrieved_docs.extend([documents[idx] for idx in top_docs_indices])
        
        print(f"Retrieved {len(all_retrieved_docs)} documents total")  # Debug print
        
        # Rerank all retrieved documents together
        rerank_results = co.rerank(
            query=query,
            documents=[doc['data']['text'] for doc in all_retrieved_docs],
            top_n=6,  # Increased to get more diverse results
            model='rerank-english-v3.0'
        )
        
        # Get the reranked documents
        reranked_documents = [all_retrieved_docs[result.index] for result in rerank_results.results]
        
        # Print sources being used
        used_sources = set(doc['data']['source'] for doc in reranked_documents)
        print(f"Using documents from sources: {used_sources}")  # Debug print
        
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
                source_doc = citation.sources[0].document
                # Make sure we have the source information
                source_pdf = source_doc.get('source', 'unknown')
                section_context = source_doc['text'][:100] + "..."  # Increased preview length
                
                # Format the citation with proper structure
                citation_source = {
                    'text': citation.text,
                    'source': {  # Changed to object structure
                        'pdf': source_pdf,
                        'section': section_context
                    }
                }
                citations.append(citation_source)
        
        return jsonify({
            'answer': chat_response.message.content[0].text,
            'citations': citations
        })
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")  # Debug print
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5001)
