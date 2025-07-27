import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone, ServerlessSpec
import os
import getpass
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Initialize Pinecone client
def initialize_pinecone():
    """Initialize Pinecone client with API key"""
    if not os.getenv("PINECONE_API_KEY"):
        os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")
    
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    return Pinecone(api_key=pinecone_api_key)

def setup_vector_store(index_name: str = "contextual-rag-obama-text-index",embeddings: str = ""):
    """Setup Pinecone vector store with your existing index"""
    pc = initialize_pinecone()
    
    # Connect to your existing index
    index = pc.Index(index_name)
    if embeddings:
        embeddings=embeddings
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        
    return PineconeVectorStore(index=index, embedding=embeddings)

def create_pc_index(pc: Pinecone, index_name: str):
    """
    Create a Pinecone vector index if it doesn't already exist.
    
    Args:
        pc (Pinecone): The Pinecone client instance.
        index_name (str): The name of the index to create.
        
    Returns:
        None
    """
    existing_indexes = [
        index_info["name"] for index_info in pc.list_indexes()
    ]

    # check if index already exists (it shouldn't if this is first time)
    if index_name not in existing_indexes:
        # if does not exist, create index
        pc.create_index(
            index_name,
            dimension=768,  # dimensionality of minilm
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        # wait for index to be initialized
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)


def create_chunks(doc_to_chunk):
    """
    Split a document into smaller chunks for processing.
    
    Args:
        doc_to_chunk: The document to be split into chunks.
        
    Returns:
        list: A list of document chunks created by the text splitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
        )
    return text_splitter.split_documents(doc_to_chunk)


def load_pdf(path):
    """
    Load a PDF file from the specified path.
    
    Args:
        path (str): The file path to the PDF document.
        
    Returns:
        list: A list of document pages loaded from the PDF.
    """
    loader = PyPDFLoader(path)
    return loader.load()


def load_chunk_file(path):
    """
    Load a PDF file and split it into chunks.
    
    This function combines the load_pdf and create_chunks functions to load a PDF
    file and then split it into smaller chunks for processing.
    
    Args:
        path (str): The file path to the PDF document.
        
    Returns:
        list: A list of document chunks created from the PDF.
    """
    doc = load_pdf(path)
    return create_chunks(doc)


def insert_data_to_index(file_path: str):
    """
    Load a PDF file and split it into chunks for processing.
    
    This function loads a PDF file from the specified path and splits it into
    smaller chunks suitable for further processing or indexing.
    
    Args:
        file_path (str): The file path to the PDF document to be processed.
        
    Returns:
        list: A list of document chunks created from the PDF.
    """
    return load_chunk_file(file_path)