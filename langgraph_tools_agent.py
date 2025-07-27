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

from util import create_pc_index, load_chunk_file,initialize_pinecone,setup_vector_store

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

 #Function to retrieve data from index
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

# Augment the LLM with tools
tools = [get_temperature,get_currency_exchange_rates,get_stock_price,youtube,retrieve_obama_speech_context,
         helper_func,retrieve_apple_10k_context,ingest_apple_10k_docs_into_vector_store]

available_functions = {
    "get_temperature": get_temperature,
    "get_currency_exchange_rates": get_currency_exchange_rates,
    "get_stock_price": get_stock_price,
    "youtube": youtube,
    "retrieve_obama_speech_context": retrieve_obama_speech_context,
    "helper_func": helper_func,
    "retrieve_apple_10k_context": retrieve_apple_10k_context
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