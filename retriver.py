import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_retriever(index_name: str = "faiss_index", embedding_model: str = "all-MiniLM-L6-v2"):
    if not os.path.exists(index_name):
        raise FileNotFoundError(f"FAISS index '{index_name}' not found. Please run upload.py first.")
    
    print(f"Loading FAISS index from '{index_name}'...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = FAISS.load_local(
        index_name, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("FAISS index loaded successfully!\n")
    
    return vector_store, embeddings

def retrieve(vector_store, query: str, k: int = 3) -> list:
    print(f"Retrieving top {k} documents for: '{query}'")
    docs = vector_store.similarity_search(query, k=k)
    print(f"Retrieved {len(docs)} documents\n")
    return docs

def retrieve_with_scores(vector_store, query: str, k: int = 3) -> list:
    print(f"Retrieving top {k} documents with scores for: '{query}'")
    docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
    print(f"Retrieved {len(docs_with_scores)} documents\n")
    return docs_with_scores

def retrieve_by_source(vector_store, query: str, source_filter: str, k: int = 3) -> list:
    print(f"Retrieving top {k} documents from source containing '{source_filter}'")
    all_docs = vector_store.similarity_search(query, k=k*3)
    filtered_docs = [
        doc for doc in all_docs 
        if source_filter.lower() in doc.metadata.get('source', '').lower()
    ][:k]
    print(f"Retrieved {len(filtered_docs)} documents after filtering\n")
    return filtered_docs

def format_retrieved_docs(docs: list, show_scores: bool = False, scores: list = None) -> str:
    formatted_text = ""
    
    for i, doc in enumerate(docs):
        formatted_text += f"\n{'='*60}\n"
        formatted_text += f"DOCUMENT {i+1}\n"
        formatted_text += f"{'='*60}\n"
        
        if doc.metadata:
            formatted_text += f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        
        if show_scores and scores and i < len(scores):
            formatted_text += f"Similarity Score: {scores[i]:.4f}\n"
        
        formatted_text += f"\nContent:\n{doc.page_content[:500]}"
        if len(doc.page_content) > 500:
            formatted_text += "...\n[Content truncated]"
        formatted_text += "\n"
    
    return formatted_text

def main():
    vector_store, embeddings = load_retriever(
        index_name="faiss_index",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    while True:
        print("\n" + "="*60)
        print("RAG RETRIEVAL SYSTEM")
        print("="*60)
        print("1. Basic retrieval")
        print("2. Retrieval with scores")
        print("3. Filter by source")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "4":
            print("Goodbye!")
            break
        
        if choice not in ["1", "2", "3"]:
            print("Invalid choice. Please try again.")
            continue
        
        query = input("Enter your search query: ").strip()
        if not query:
            print("Query cannot be empty!")
            continue
        
        k = input("Number of documents to retrieve (default: 3): ").strip()
        k = int(k) if k.isdigit() else 3
        
        if choice == "1":
            docs = retrieve(vector_store, query, k=k)
            print(format_retrieved_docs(docs))
            
        elif choice == "2":
            docs_with_scores = retrieve_with_scores(vector_store, query, k=k)
            docs = [doc for doc, _ in docs_with_scores]
            scores = [score for _, score in docs_with_scores]
            print(format_retrieved_docs(docs, show_scores=True, scores=scores))
            
        elif choice == "3":
            source_filter = input("Enter source filter (e.g., filename or path): ").strip()
            docs = retrieve_by_source(vector_store, query, source_filter, k=k)
            print(format_retrieved_docs(docs))

if __name__ == "__main__":
    main()