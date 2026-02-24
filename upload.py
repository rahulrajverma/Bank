import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS
import fitz
import pandas as pd
from langchain_core.documents import Document

def load_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text() + "\n" 
    return text

def load_excel(file_path):
    df = pd.read_excel(file_path)
    text = df.to_string(index=False)  
    return text

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
    
def load_documents(data_folder):
    docs = []

    for file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file)

        if file.endswith(".txt"):
            text = load_text(file_path)
        elif file.endswith(".pdf"):
            text = load_pdf(file_path)
        elif file.endswith((".xlsx", ".xls")):
            text = load_excel(file_path)
        else:
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={"source": file_path}
            )
        )

    print(f"Total documents loaded: {len(docs)}\n")
    return docs

def chunk_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
        print(f"Created {len(chunks)} chunks from documents")
        chunk_lengths = [len(c.page_content) for c in chunks]
        print(f"Chunk size range: {min(chunk_lengths)} - {max(chunk_lengths)} characters\n")
    else:
        print("No chunks created - check if documents were loaded properly\n")
    
    return chunks

def create_faiss_index(chunks: list, embedding_model: str = "all-MiniLM-L6-v2", index_name: str = "faiss_index") -> FAISS:
    if not chunks:
        raise ValueError("No chunks provided to create FAISS index")
    
    print(f"Using embedding model: {embedding_model}")
    print("Generating embeddings (this may take a moment)...")
    
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    vector_store.save_local(index_name)
    print(f"FAISS index saved as '{index_name}'\n")
    
    return vector_store

def upload_data_to_faiss(
    data_folder: str = "Data",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = "all-MiniLM-L6-v2",
    index_name: str = "faiss_index"
) -> FAISS:
    
    print("Step 1: Loading documents...")
    documents = load_documents(data_folder)
    
    if not documents:
        raise ValueError(f"No documents found in folder: {data_folder}")
    
    print("Step 2: Creating chunks...")
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    
    if not chunks:
        raise ValueError("No chunks created from documents")
    
    print("Step 3: Creating FAISS index...")
    vector_store = create_faiss_index(chunks, embedding_model, index_name)
    
    print("=" * 60)
    print("RAG Data Upload Complete!")
    print("=" * 60)
    
    return vector_store

if __name__ == "__main__":
    vector_store = upload_data_to_faiss(
        data_folder="Data",
        chunk_size=1000,          
        chunk_overlap=200,        
        embedding_model="all-MiniLM-L6-v2",  
        index_name="faiss_index" 
    )