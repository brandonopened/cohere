import cohere
import numpy as np
from PyPDF2 import PdfReader

# Initialize Cohere client
co = cohere.ClientV2(api_key="VvImZNZXUXLfUq23reRIyTTlqI2SvRiQOBOwKK18")

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

def main():
    # Load and process the PDF
    print("Loading and processing PDF...")
    documents = extract_pdf_text("indianadoe.pdf")
    print(f"Total documents extracted: {len(documents)}")
    
    # Create embeddings
    print("Creating embeddings...")
    doc_embeddings = batch_embed_documents(documents)
    
    print("\nReady to answer questions! (Type 'quit' to exit)")
    
    while True:
        query = input("\nWhat would you like to know about the document? ").strip()
        
        if query.lower() == 'quit':
            break
            
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
        
        # Print response and citations
        print("\nAnswer:", chat_response.message.content[0].text)
        
        if chat_response.message.citations:
            print("\nSources:")
            for i, citation in enumerate(chat_response.message.citations, 1):
                print(f"\n{i}. {citation.text}")
                print(f"   From: {citation.sources[0].document['text'][:200]}...")

if __name__ == "__main__":
    main()