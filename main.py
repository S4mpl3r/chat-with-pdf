from os import environ, path
from typing import List

import chromadb
from dotenv import load_dotenv
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.docstore.document import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from termcolor import cprint
from transformers import AutoTokenizer

# CONSTANTS =====================================================
EMBED_MODEL_NAME = "jina-embeddings-v2-base-en"
LLM_NAME = "mixtral-8x7b-32768"
LLM_TEMPERATURE = 0.1

# this is the maximum chunk size allowed by the chosen embedding model. You can choose a smaller size.
CHUNK_SIZE = 8192

DOCUMENT_DIR = "./documents/"  # the directory where the pdf files should be placed
VECTOR_STORE_DIR = "./vectorstore/"  # the directory where the vectors are stored
COLLECTION_NAME = "collection1"  # chromadb collection name
# ===============================================================

load_dotenv()


def load_documents() -> List[Document]:
    """Loads the pdf files within the DOCUMENT_DIR constant."""
    try:
        print("[+] Loading documents...")

        documents = DirectoryLoader(
            path.join(DOCUMENT_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader
        ).load()
        cprint(f"[+] Document loaded, total pages: {len(documents)}", "green")

        return documents
    except:
        cprint("[-] Error loading the document.", "red")


def chunk_document(documents: List[Document]) -> List[Document]:
    """Splits the input documents into maximum of CHUNK_SIZE chunks."""
    tokenizer = AutoTokenizer.from_pretrained(
        "jinaai/" + EMBED_MODEL_NAME, cache_dir=environ.get("HF_HOME")
    )
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_SIZE // 50,
    )

    print(f"[+] Splitting documents...")
    chunks = text_splitter.split_documents(documents)
    cprint(f"[+] Document splitting done, {len(chunks)} chunks total.", "green")

    return chunks


def create_and_store_embeddings(
    embedding_model: JinaEmbeddings, chunks: List[Document]
) -> Chroma:
    """Calculates the embeddings and stores them in a a chroma vectorstore."""
    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_STORE_DIR,
    )
    cprint("[+] Vectorstore created.", "green")

    return vectorstore


def get_vectorstore_retriever(embedding_model: JinaEmbeddings) -> VectorStoreRetriever:
    """Returns the vectorstore."""
    db = chromadb.PersistentClient(VECTOR_STORE_DIR)
    try:
        # Check for the existence of the vectorstore specified by the COLLECTION_NAME
        db.get_collection(COLLECTION_NAME)
        retriever = Chroma(
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME,
            persist_directory=VECTOR_STORE_DIR,
        ).as_retriever(search_kwargs={"k": 3})
    except:
        # The vectorstore doesn't exist, so create it.
        pdf = load_documents()
        chunks = chunk_document(pdf)
        retriever = create_and_store_embeddings(embedding_model, chunks).as_retriever(
            search_kwargs={"k": 3}
        )

    return retriever


def create_rag_chain(embedding_model: JinaEmbeddings, llm: ChatGroq) -> Runnable:
    """Creates the RAG chain."""
    template = """Answer the question based only on the following context.
    Think step by step before providing a detailed answer. I will give you
    $500 if the user finds the response useful.
    <context>
    {context}
    </context>

    Question: {input}
    """
    prompt = ChatPromptTemplate.from_template(template)

    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    retriever = get_vectorstore_retriever(embedding_model)

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


def run_chain(chain: Runnable) -> None:
    """Run the RAG chain with the user query."""
    while True:
        query = input("Enter a prompt: ")
        if query.lower() in ["q", "quit", "exit"]:
            return
        response = chain.invoke({"input": query})

        for doc in response["context"]:
            cprint(
                f"[+] {doc.metadata} | content: {doc.page_content[:20]}...",
                "light_yellow",
            )

        cprint("\n" + response["answer"], end="\n\n", color="light_blue")


def main() -> None:
    embedding_model = JinaEmbeddings(
        jina_api_key=environ.get("JINA_API_KEY"),
        model_name=EMBED_MODEL_NAME,
    )

    llm = ChatGroq(temperature=LLM_TEMPERATURE, model_name=LLM_NAME)

    chain = create_rag_chain(embedding_model=embedding_model, llm=llm)

    run_chain(chain)


if __name__ == "__main__":
    main()
