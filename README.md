# Chat With PDFs
Chat with your PDF files for free, using [Langchain](https://python.langchain.com/docs/get_started/quickstart), [Groq](https://console.groq.com/), [Chroma](https://docs.trychroma.com/getting-started) vector store, and [Jina AI](https://jina.ai/embeddings/) embeddings. This repository contains a simple Python implementation of the RAG (Retrieval-Augmented-Generation) system. The RAG model is used to retrieve relevant chunks of the user PDF file based on user queries and provide informative responses.

## Installation
Follow these steps:
1. Clone the repository
   ```
   git clone https://github.com/S4mpl3r/chat-with-pdf.git
   ```
2. Create a virtual environment and activate it (optional, but highly recommended).
   ```
   python -m venv .venv
   Windows: .venv\Scripts\activate
   Linux: source .venv/bin/activate
   ```
3. Install required packages:
   ```
   python -m pip install -r requirements.txt
   ```
4. Create a .env file in the root of the project and populate it with the following keys. You'll need to obtain your api keys:
   ```
   JINA_API_KEY=<YOUR KEY>
   GROQ_API_KEY=<YOUR KEY>
   HF_TOKEN=<YOUR TOKEN>
   HF_HOME=<PATH TO STORE HUGGINGFACE MODEL>
   ```
5. Run the program:
   ```
   python main.py
   ```
## Configuration
You can customize the behavior of the system by modifying the constants and parameters in the main.py file:

* EMBED_MODEL_NAME: Specify the name of the Jina embedding model to be used.
* LLM_NAME: Specify the name of the language model (Refer to [Groq](https://groq.com/) for the list of available models).
* LLM_TEMPERATURE: Set the temperature parameter for the language model.
* CHUNK_SIZE: Specify the maximum chunk size allowed by the embedding model.
* DOCUMENT_DIR: Specify the directory where PDF documents are stored.
* VECTOR_STORE_DIR: Specify the directory where vector embeddings are stored.
* COLLECTION_NAME: Specify the name of the collection for the chroma vector store.

## Resources
Kudos to the amazing libraries and services listed below:
* [Langchain](https://www.langchain.com/)
* [Groq](https://groq.com/)
* [Jina AI](https://jina.ai/)
* [ChromaDB](https://www.trychroma.com/)

## License
MIT



