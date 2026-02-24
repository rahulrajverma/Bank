import os
from retriver import load_retriever, retrieve
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def setup_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set GEMINI_API_KEY in .env file")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3
    )
    return llm

def get_answer(llm, query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""Answer the question based on this context only.
If answer not in context, say "I don't know".

Context: {context}

Question: {query}

Answer: """
    
    response = llm.invoke(prompt)
    return response.content

def main():
    print("Loading retriever...")
    vector_store, _ = load_retriever()
    
    print("Loading Gemini...")
    llm = setup_gemini()
    
    while True:
        print("\n" + "-"*50)
        query = input("Enter your question (or 'quit' to exit): ").strip()
        
        if query.lower() == 'quit':
            break
        
        if not query:
            continue
        
        k = input("How many documents to retrieve? (default 3): ").strip()
        k = int(k) if k.isdigit() else 3
        
        print("\nSearching...")
        docs = retrieve(vector_store, query, k)
        
        print("\nGenerating answer...")
        answer = get_answer(llm, query, docs)
        
        print("\n" + "="*50)
        print(f"Q: {query}")
        print(f"A: {answer}")
        print("="*50)
        
        print("\nSources used:")
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc.metadata.get('source', 'Unknown')}")

if __name__ == "__main__":
    main()