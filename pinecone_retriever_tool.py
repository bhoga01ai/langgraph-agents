import os
import getpass
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool

from dotenv import load_dotenv
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# Initialize Pinecone client
def initialize_pinecone():
    """Initialize Pinecone client with API key"""
    if not os.getenv("PINECONE_API_KEY"):
        os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")
    
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    return Pinecone(api_key=pinecone_api_key)

def setup_vector_store(index_name: str = "contextual-rag-obama-text-index"):
    """Setup Pinecone vector store with your existing index"""
    pc = initialize_pinecone()
    
    # Connect to your existing index
    index = pc.Index(index_name)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        
    return PineconeVectorStore(index=index, embedding=embeddings)

@tool
def retrieve_obama_context(
    query: str, 
    k: int = 5, 
    score_threshold: float = 0.7
) -> str:
    """
    Retrieve relevant context from your Obama text Pinecone vector store for a given query.
    
    Args:
        query: The user's query to search for relevant Obama-related context
        k: Number of top results to return (default: 3)
        score_threshold: Minimum similarity score threshold (default: 0.7)
    
    Returns:
        Formatted string containing the retrieved Obama text context
    """
    try:
        # Setup vector store with your existing index
        vector_store = setup_vector_store("semantic-search-obama-text-may2025")
        retriever = vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": k, "score_threshold": score_threshold},
                  )
        results = retriever.invoke(query)
        context=''
        for result in results:
            context += result.page_content
        return context
    except Exception as e:
        return f"Error retrieving context: {str(e)}"
        
# Example usage function
def example_usage():
    """
    Example of how to use the Obama text retrieval tools
    """
    # Search for context about healthcare - USE .invoke() method
    context= retrieve_obama_context.invoke({
        "query": "what was obama said about schools",
        "k": 5,
        "score_threshold": 0.7
    })
    print(context)

if __name__ == "__main__":
    example_usage()