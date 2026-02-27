# Task 1

- step 1 -: create a .env file and give the gemni api key like this
   GEMINI_API_KEY=your_gemini_api_key_here

- step 2 -: create a DATA folder and put some text, pdf, or excel files in it to test the system

- step 3 -: create a virtual environment and install the required    libraries from requirements.txt
            ```pip install -r requirement.txt```

- step 4 -: run upload.py to create the FAISS index from your documents
            ```python ./upload.py```
- step 5 -: run retriever.py to test the retrieval of documents based on a query
            ```python ./retriver.py```
- step 6 -: run generation.py to test the end-to-end RAG system with Gemini LLM

- step 7 -: run main.py to launch the Streamlit app and interact with the RAG system through a web interface
            ```streamlit run main.py```
- step 8 -: improve the prompt for generating the logic

- step 9 -: update stremlit 


# Task 2

- Create a history of the querry which we asked
