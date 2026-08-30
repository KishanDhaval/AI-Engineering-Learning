import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from Week_4 import config
from Week_4.llm import cloud_embeddings, local_embeddings

def get_vectorstore(use_local: bool = False, force_reindex: bool = False):
    """Load FAISS vector store from disk, or create a new one from the PDF."""
    db_path = config.FAISS_LOCAL_PATH if use_local else config.FAISS_CLOUD_PATH
    embeddings = local_embeddings if use_local else cloud_embeddings

    if os.path.exists(db_path) and not force_reindex:
        return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

    print(f"Indexing PDF into FAISS ({'Local' if use_local else 'Cloud'})...")
    loader = PyPDFLoader(config.DATA_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(db_path)
    return vectorstore
