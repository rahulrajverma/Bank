import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        "faiss_index", 
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store

def get_answer(vector_store, query, k=3):
    docs = vector_store.similarity_search(query, k=k)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Please set GEMINI_API_KEY in .env file", []
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3
    )
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""Answer the question based on this context only.
If answer not in context, say "I don't know".

Context: {context}

Question: {query}

Answer: """
    
    response = llm.invoke(prompt)
    return response.content, docs

def main():
    
    st.set_page_config(page_title="RAG Q&A System", page_icon="❓")
    
    st.title("RAG Question Answering System")
    st.markdown("Ask questions based on your indexed documents")
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    try:
        vector_store = load_vector_store()
        st.sidebar.success("FAISS index loaded successfully")
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        st.info("Please make sure 'faiss_index' folder exists in the current directory")
        return
    
    with st.sidebar:
        st.header("⚙️ Settings")
        k = st.slider("Number of documents to retrieve", min_value=1, max_value=5, value=3)
        st.header("🕘 Query History")

        for i, q in enumerate(st.session_state.history, 1):
            st.write(f"{i}. {q}")
    
    query = st.text_input("Enter your question:", placeholder="What would you like to know?")
    
    if query:
        if st.button("Get Answer", type="primary"):
            st.session_state.history.append(query)
            with st.spinner("Searching and generating answer..."):
                answer, docs = get_answer(vector_store, query, k)
            
            st.markdown("### Answer")
            st.markdown(answer)
            
            with st.expander("View source documents"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Source {i}:** {doc.metadata.get('source', 'Unknown')}")
                    st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    st.divider()

if __name__ == "__main__":
    main()