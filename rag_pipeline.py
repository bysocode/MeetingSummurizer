from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import os

def build_rag_index(data_dir: str, llm_model: str = "llama3.2:latest") -> VectorStoreIndex:
    """
    Build a RAG (Retrieval-Augmented Generation) index using HuggingFace embeddings and Ollama LLM.

    Args:
        data_dir (str): Path to the directory containing text documents or summaries in `.txt` format.
        llm_model (str): The model name for the Ollama LLM. Default is "llama3.1:8b".

    Returns:
        VectorStoreIndex: Built index for querying.
    """
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"No documents found in directory: {data_dir}. Please add documents to enable insights.")
        return None

    print("Loading documents from the directory...")
    loader = SimpleDirectoryReader(data_dir)
    documents = loader.load_data()

    if not documents:
        print("No valid documents found in the directory.")
        return None

    print("Building the index with HuggingFaceEmbedding and Ollama LLM...")
    embedder = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    ollama_llm = Ollama(model=llm_model)

    # Build the VectorStoreIndex
    index = VectorStoreIndex.from_documents(documents, embed_model=embedder, llm=ollama_llm)
    print("VectorStoreIndex created successfully.")
    return index


def query_rag_index(index: VectorStoreIndex, query: str, llm_model: str = "llama3.2:latest") -> str:
    """
    Query the RAG index for insights.

    Args:
        index (VectorStoreIndex): The index to query.
        query (str): The query string to search for insights.
        llm_model (str): The model name for the Ollama LLM.

    Returns:
        str: Query result from the index.
    """
    if index is None:
        return "Cannot query the index as it has not been built. Please add documents and rebuild the index."

    print("Creating query engine...")
    ollama_llm = Ollama(model=llm_model)
    query_engine = index.as_query_engine(llm=ollama_llm)

    print("Running query...")
    try:
        response = query_engine.query(query)
        return response
    except Exception as e:
        return f"Error during query execution: {e}"
