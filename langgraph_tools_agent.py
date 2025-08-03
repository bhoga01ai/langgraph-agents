#   LangGraph Tools agent

# Architecture 
# 1. Define a lLM 
# 2. Define a State - MesasgeState
# 3. Define a Tools -get_temperature,get_currency_exchange_rates,get_stock_price,retrieve_obama_speech_context, youtube_search
# 4. Define an Agent - React Agent
# 5. Invoke the Agent

# STEP 0 - Import ENVIRONMENT VAIRABLES and Standard libraries
import os
import re
import select
import yfinance as yf
import requests

from dotenv import load_dotenv
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
weatherAPIKey = str(os.getenv('weatherAPIKey'))

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.tools import tool
from IPython.display import Image, display
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_community.tools import YouTubeSearchTool
from langchain_pinecone import PineconeVectorStore

from util import create_pc_index, load_chunk_file,initialize_pinecone,setup_vector_store,process_uploaded_file

from langchain_huggingface import HuggingFaceEmbeddings

pc_apple_index_name = 'langgragh-tools-apple-pc-index'   # this is a new index

pc_obama_index_name="semantic-search-obama-text-may2025"   # This is a existing index

# Define embedding model
embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name = embedding_model_name)

# STEP 1: Define the LLM
llm = init_chat_model(model="gemini-2.0-flash",model_provider="google_genai",temperature=0.5)
# a Simple test 
# response=llm.invoke("who are you")
# print(response.content)

# STEP 2: Define tools
# Function to get temperature
@tool
def get_temperature(city: str) -> dict:
    """Gets the current temperature for a given city.

    Args:
        city (str): The name of the city (e.g., 'San Francisco').

    Returns:
        dict: A dictionary containing the temperature data or an error message.
    """
    print("Entered the method / function get_temperature");
    weatherAPIUrl = "http://api.weatherapi.com/v1/current.json?key=" + weatherAPIKey + "&q=" + city;
    print(weatherAPIUrl)
    response = requests.get(weatherAPIUrl)
    data = response.json()
    print(data)
    return data

@tool
# Function to get currency exchange rates
def get_currency_exchange_rates(currency: str) -> dict:
    """Gets the currency exchange rates for a given currency.

    Args:
        currency (str): The currency code (e.g., 'USD').

    Returns:
        dict: A dictionary containing the exchange rate data.
    """
    print("Entered the method / function get_currency_exchange_rates");
    # Where USD is the base currency you want to use
    url = 'https://v6.exchangerate-api.com/v6/6f9f5f76947ce2150d20b85c/latest/' + currency + "/"

    # Making our request
    response = requests.get(url)
    data = response.json()
    return data

@tool
# Function to Get Stock Price
def get_stock_price(ticker: str) -> dict:
    """Gets the stock price for a given ticker symbol.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple).

    Returns:
        dict: A dictionary containing the stock price or an error message.
    """
    print("Entered the method / function get_stock_price");
    print(ticker)
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    if not hist.empty:
        return {"price": str(hist['Close'].iloc[-1])}
    else:
        return {"error": "No data available"}


youtube = YouTubeSearchTool(
   description="A tool to search YouTube videos. Use this tool if you think the user’s asked concept can be best explained by watching a video."
)


@tool
def retrieve_obama_speech_context(
    query: str, 
    k: int = 5, 
    score_threshold: float = 0.7,
    index_name: str = pc_obama_index_name 
) -> str:
    """
    Retrieve relevant context from your Obama text Pinecone vector store for a given query.
    
    Args:
        query: The user's query to search for relevant Obama-related context
        k: Number of top results to return (default: 5)
        score_threshold: Minimum similarity score threshold (default: 0.7)
    
    Returns:
        Formatted string containing the retrieved Obama text context
    """
    try:
        # Setup vector store with your existing index
        vector_store = setup_vector_store(index_name)
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

@tool
# Function to insert data to index
def ingest_apple_10k_docs_into_vector_store(file_path: str,index_name: str = pc_apple_index_name):
    """
    Ingest Apple 10-K document data into a Pinecone vector store for semantic search.
    
    This function processes a PDF file containing Apple's 10-K financial report,
    chunks the content into manageable pieces, converts them to vector embeddings,
    and stores them in a Pinecone vector database for later retrieval.
    
    Args:
        file_path (str): The absolute or relative path to the Apple 10-K PDF file
                        to be processed and ingested into the vector store.
        index_name (str): The name of the Pinecone vector index where the data will be stored.
                         Default is 'langgragh-tools-apple-pc-index'.
                        
    Returns:
        str: A success message indicating that the document data has been
             successfully processed and inserted into the Pinecone vector store.
             
    """
    
    # Initialize pinecone client
    pc = initialize_pinecone()
    # create index
    create_pc_index(pc, index_name)
    # load and chunk data
    chunks = load_chunk_file(file_path)
    # insert/upsert data
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
    return 'Data inserted successfully'

@tool
def retrieve_apple_10k_context(
    query: str, 
    k: int = 5, 
    score_threshold: float = 0.7,
    index_name: str = pc_apple_index_name
) -> str:
    """
    Retrieve relevant context from your Apple 10k Pinecone vector store for a given query.
    
    Args:
        query: The user's query to search for relevant Obama-related context
        k: Number of top results to return (default: 5)
        score_threshold: Minimum similarity score threshold (default: 0.7)
    
    Returns:
        Formatted string containing the retrieved Apple 10k text context
    """
    try:
        # Setup vector store with your existing index
        vector_store = setup_vector_store(index_name,embeddings)
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


@tool
def drop_pinecone_index(index_name: str) -> str:
    """
    Drop/delete a Pinecone vector index by name.
    
    This function connects to Pinecone and deletes the specified index.
    Use this tool when you need to remove an index that is no longer needed
    or to clean up resources.
    
    Args:
        index_name (str): The name of the Pinecone index to be deleted.
                         Example: 'langgragh-tools-apple-pc-index' or 'semantic-search-obama-text-may2025'
    
    Returns:
        str: A success message if the index was deleted successfully,
             or an error message if the operation failed.
             
    Raises:
        Exception: If there's an error connecting to Pinecone or deleting the index.
    """
    try:
        # Initialize pinecone client
        pc = initialize_pinecone()
        
        # Check if index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            return f"Index '{index_name}' does not exist. Available indexes: {existing_indexes}"
        
        # Delete the index
        pc.delete_index(index_name)
        
        # Verify deletion
        remaining_indexes = [index.name for index in pc.list_indexes()]
        
        if index_name not in remaining_indexes:
            return f"Index '{index_name}' has been successfully deleted. Remaining indexes: {remaining_indexes}"
        else:
            return f"Failed to delete index '{index_name}'. It may still exist."
            
    except Exception as e:
        return f"Error dropping index '{index_name}': {str(e)}"

@tool
def list_pinecone_indexes() -> str:
    """
    List all available Pinecone vector indexes.
    
    This function connects to Pinecone and retrieves a list of all existing indexes
    in your Pinecone environment. Useful for checking what indexes are available
    before performing operations.
    
    Returns:
        str: A formatted string containing all available index names and their details,
             or an error message if the operation failed.
    """
    try:
        # Initialize pinecone client
        pc = initialize_pinecone()
        
        # Get list of indexes
        indexes = pc.list_indexes()
        
        if not indexes:
            return "No Pinecone indexes found in your environment."
        
        index_info = "Available Pinecone Indexes:\n\n"
        for i, index in enumerate(indexes, 1):
            index_info += f"{i}. Name: {index.name}\n"
            index_info += f"   Dimension: {index.dimension}\n"
            index_info += f"   Metric: {index.metric}\n"
            index_info += f"   Host: {index.host}\n\n"
        
        return index_info
        
    except Exception as e:
        return f"Error listing Pinecone indexes: {str(e)}"

@tool   # Function to retrieve all tools 
def helper_func():
    """
    Retrieve information about all available tools in the system.
    
    This function uses the docstrings of each function to provide information about
    all available tools, including their names, descriptions, and the total count.
    
    Args:
        None
        
    Returns:
        str: A formatted string containing the number of tools and information about each tool,
             including their names and descriptions from docstrings.
    """
    tool_info_str = ""
    
    # Get the number of tools
    num_tools = len(available_functions)
    tool_info_str += f"Number of available tools: {num_tools}\n\n"
    
    # Iterate through each function and get its docstring
    for tool_name, tool_func in available_functions.items():
        # Get the docstring of the function
        doc = tool_func.__doc__
        
        # Extract the first line of the docstring as a short description
        description = "No description available"
        if doc:
            # Split the docstring by lines and get the first non-empty line
            doc_lines = [line.strip() for line in doc.split('\n') if line.strip()]
            if doc_lines:
                description = doc_lines[0]
        
        # Add tool information to the result string
        tool_info_str += f"Tool: {tool_name}\nDescription: {description}\n\n"
    
    return tool_info_str


@tool
def handle_file_upload(attachment_data: str) -> str:
    """
    Handle file uploads from LangStudio interface.
    
    This tool processes file attachments sent from the LangStudio interface
    and extracts the file path or content for further processing.
    
    Args:
        attachment_data (str): String representation of file attachment information
                              that will be parsed into a dictionary, or just a filename
        
    Returns:
        str: File path or processing result
    """
    import json
    import ast
    import os
    
    try:
        # First, check if it's just a simple filename
        if not attachment_data.startswith('{') and not attachment_data.startswith('['):
            # It's likely just a filename, treat it as a file path
            filename = attachment_data.strip()
            
            # Try different possible locations for the file
            possible_paths = [
                filename,  # Current directory
                os.path.join(os.getcwd(), filename),  # Explicit current directory
                os.path.join(os.getcwd(), 'docs', filename),  # docs folder
                os.path.join(os.getcwd(), 'docs', 'apple_10k.pdf'),  # Default Apple 10K file
            ]
            
            for file_path in possible_paths:
                if os.path.exists(file_path):
                    result = ingest_apple_10k_docs_into_vector_store(file_path)
                    return f"Successfully processed file '{filename}' from path '{file_path}': {result}"
            
            # If no file found, provide helpful message
            return f"File '{filename}' not found. Tried locations: {', '.join(possible_paths)}. Please ensure the file exists or use the full file path."
        
        # Try to parse the string as JSON first
        try:
            data = json.loads(attachment_data)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to evaluate as Python literal
            try:
                data = ast.literal_eval(attachment_data)
            except (ValueError, SyntaxError):
                return f"Error: Could not parse attachment data as JSON or Python literal: {attachment_data}"
        
        if not isinstance(data, dict):
            return f"Error: Parsed data is not a dictionary: {type(data)}"
            
        if 'path' in data:
            # If file path is directly available
            file_path = data['path']
            filename = data.get('name', 'uploaded_file')
            
            # Convert to absolute path if it's relative
            if not os.path.isabs(file_path):
                # Check if file exists in current directory
                current_dir_path = os.path.join(os.getcwd(), file_path)
                if os.path.exists(current_dir_path):
                    file_path = current_dir_path
                else:
                    # Check if it's just a filename, look in docs folder
                    docs_path = os.path.join(os.getcwd(), 'docs', file_path)
                    if os.path.exists(docs_path):
                        file_path = docs_path
                    else:
                        return f"Error: File not found. Tried paths: {current_dir_path}, {docs_path}"
            
            # Verify file exists
            if not os.path.exists(file_path):
                return f"Error: File does not exist at path: {file_path}"
            
            # Process the file
            result = ingest_apple_10k_docs_into_vector_store(file_path)
            return f"Successfully processed file '{filename}' from path '{file_path}': {result}"
            
        elif 'content' in data:
            # If file content is provided (base64 encoded)
            filename = data.get('name', 'uploaded_file')
            file_type = data.get('type', 'pdf').lower().replace('application/', '')
            content = data['content']
            
            return process_uploaded_file(content, filename, file_type)
            
        else:
            return f"Invalid attachment data. Expected 'path' or 'content' key. Received: {list(data.keys())}"
            
    except Exception as e:
        return f"Error handling file upload: {str(e)}"

@tool
def get_pinecone_index_details(index_name: str) -> str:
    """
    Get comprehensive details about a specific Pinecone index including embedding model info and metadata.
    
    This function retrieves detailed information about a Pinecone index including:
    - Basic index properties (name, dimension, metric, host)
    - Index statistics (vector count, index size)
    - Metadata information
    - Configuration details
    - Status and readiness
    
    Args:
        index_name (str): The name of the Pinecone index to get details for.
                         Example: 'langgragh-tools-apple-pc-index' or 'semantic-search-obama-text-may2025'
    
    Returns:
        str: A formatted string containing comprehensive index details,
             or an error message if the operation failed.
    """
    try:
        # Initialize pinecone client
        pc = initialize_pinecone()
        
        # Check if index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            return f"Index '{index_name}' does not exist. Available indexes: {existing_indexes}"
        
        # Get index details from list_indexes (basic info)
        index_info = None
        for index in pc.list_indexes():
            if index.name == index_name:
                index_info = index
                break
        
        if not index_info:
            return f"Could not retrieve information for index '{index_name}'"
        
        # Connect to the index to get statistics
        index = pc.Index(index_name)
        
        # Get index statistics
        try:
            stats = index.describe_index_stats()
        except Exception as stats_error:
            stats = None
            stats_error_msg = str(stats_error)
        
        # Format the comprehensive details
        details = f"📊 Pinecone Index Details: {index_name}\n"
        details += "=" * 50 + "\n\n"
        
        # Basic Information
        details += "🔧 Basic Information:\n"
        details += f"   • Name: {index_info.name}\n"
        details += f"   • Dimension: {index_info.dimension}\n"
        details += f"   • Metric: {index_info.metric}\n"
        details += f"   • Host: {index_info.host}\n"
        
        # Try to get additional properties if available
        if hasattr(index_info, 'status'):
            details += f"   • Status: {index_info.status}\n"
        if hasattr(index_info, 'spec'):
            details += f"   • Spec: {index_info.spec}\n"
        
        details += "\n"
        
        # Statistics Information
        if stats:
            details += "📈 Index Statistics:\n"
            if 'total_vector_count' in stats:
                details += f"   • Total Vectors: {stats['total_vector_count']:,}\n"
            
            if 'dimension' in stats:
                details += f"   • Vector Dimension: {stats['dimension']}\n"
            
            if 'index_fullness' in stats:
                fullness_percent = stats['index_fullness'] * 100
                details += f"   • Index Fullness: {fullness_percent:.2f}%\n"
            
            # Namespace information
            if 'namespaces' in stats and stats['namespaces']:
                details += f"   • Namespaces: {len(stats['namespaces'])}\n"
                for namespace, ns_stats in stats['namespaces'].items():
                    ns_name = namespace if namespace else "(default)"
                    vector_count = ns_stats.get('vector_count', 0)
                    details += f"     - {ns_name}: {vector_count:,} vectors\n"
            else:
                details += "   • Namespaces: None (using default namespace)\n"
        else:
            details += "📈 Index Statistics:\n"
            details += f"   • Error retrieving stats: {stats_error_msg if 'stats_error_msg' in locals() else 'Unknown error'}\n"
        
        details += "\n"
        
        # Embedding Model Information (inferred from dimension)
        details += "🤖 Embedding Model Information:\n"
        dimension = index_info.dimension
        
        # Common embedding model dimensions
        model_mapping = {
            384: "sentence-transformers/all-MiniLM-L6-v2 or similar 384-dim model",
            512: "sentence-transformers/all-mpnet-base-v2 or similar 512-dim model", 
            768: "sentence-transformers/all-mpnet-base-v2, BERT-base, or similar 768-dim model",
            1024: "OpenAI text-embedding-ada-002 (legacy) or similar 1024-dim model",
            1536: "OpenAI text-embedding-ada-002 or text-embedding-3-small",
            3072: "OpenAI text-embedding-3-large",
            4096: "Cohere embed-english-v3.0 or similar large model"
        }
        
        if dimension in model_mapping:
            details += f"   • Likely Model: {model_mapping[dimension]}\n"
        else:
            details += f"   • Custom Model: {dimension}-dimensional embedding model\n"
        
        details += f"   • Vector Dimension: {dimension}\n"
        details += f"   • Distance Metric: {index_info.metric}\n"
        
        # Recommendations based on metric
        if index_info.metric == 'cosine':
            details += "   • Metric Note: Cosine similarity - good for text embeddings\n"
        elif index_info.metric == 'euclidean':
            details += "   • Metric Note: Euclidean distance - good for normalized vectors\n"
        elif index_info.metric == 'dotproduct':
            details += "   • Metric Note: Dot product - good for normalized vectors\n"
        
        details += "\n"
        
        # Usage Recommendations
        details += "💡 Usage Recommendations:\n"
        if stats and 'total_vector_count' in stats:
            vector_count = stats['total_vector_count']
            if vector_count == 0:
                details += "   • Index is empty - ready for data ingestion\n"
            elif vector_count < 1000:
                details += "   • Small index - suitable for testing and development\n"
            elif vector_count < 100000:
                details += "   • Medium index - good for production use cases\n"
            else:
                details += "   • Large index - enterprise-scale deployment\n"
        
        details += f"   • Optimal for: {dimension}-dimensional vector similarity search\n"
        details += f"   • Best practices: Use {index_info.metric} distance for queries\n"
        
        return details
        
    except Exception as e:
        return f"Error getting details for index '{index_name}': {str(e)}"

# Update the tools list
tools = [get_temperature,get_currency_exchange_rates,get_stock_price,youtube,retrieve_obama_speech_context,
         helper_func,retrieve_apple_10k_context,ingest_apple_10k_docs_into_vector_store,
         drop_pinecone_index,list_pinecone_indexes,get_pinecone_index_details,handle_file_upload]

# Update the available_functions dictionary
available_functions = {
    "get_temperature": get_temperature,
    "get_currency_exchange_rates": get_currency_exchange_rates,
    "get_stock_price": get_stock_price,
    "youtube": youtube,
    "retrieve_obama_speech_context": retrieve_obama_speech_context,
    "helper_func": helper_func,
    "retrieve_apple_10k_context": retrieve_apple_10k_context,
    "ingest_apple_10k_docs_into_vector_store": ingest_apple_10k_docs_into_vector_store,
    "drop_pinecone_index": drop_pinecone_index,
    "list_pinecone_indexes": list_pinecone_indexes,
    "get_pinecone_index_details": get_pinecone_index_details,
    "handle_file_upload": handle_file_upload,
}
# Create a dictionary of tools by name
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

print(tools_by_name)

# STEP 3: Define an Agent -React 
# Pass in:
# (1) the augmented LLM with tools
# (2) the tools list (which is used to create the tool node)
pre_built_agent = create_react_agent(llm_with_tools, tools=tools,name="langgraph_tools_agent")

# Show the agent
# To save as a PNG image file:
png_data = pre_built_agent.get_graph().draw_mermaid_png()
with open("pre_built_agent_graph.png", "wb") as f:
    f.write(png_data)
print("Graph saved as workflow_graph.png")

# To save as a Mermaid diagram file (.mmd):
mermaid_text = pre_built_agent.get_graph().draw_mermaid()
with open("pre_built_agent_graph.mmd", "w") as f:
    f.write(mermaid_text)
print("Graph saved as workflow_graph.mmd")
# To display it in a notebook, you can still use:
display(Image(png_data))

# Invoke
messages = [HumanMessage(content="what was obama said about elementary schools and explain about stock exchange")]
messages = pre_built_agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()

