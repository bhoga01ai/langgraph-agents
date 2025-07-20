#   LangGraph Tools agent

# Architecture 
# 1. Define a lLM 
# 2. Define a State - MesasgeState
# 3. Define a Tools
# 4. Define an Agent

# STEP 0 - Import ENVIRONMENT VAIRABLES and Standard libraries
import os
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

# STEP 1: Define the LLM
llm = init_chat_model(model="gemini-2.5-flash",model_provider="google_genai",temperature=0.5)
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

# Augment the LLM with tools
tools = [get_temperature,get_currency_exchange_rates,get_stock_price,youtube]
# Create a dictionary of tools by name
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

print(tools_by_name)


# STEP 3: Define an Agent -React 
# Pass in:
# (1) the augmented LLM with tools
# (2) the tools list (which is used to create the tool node)
pre_built_agent = create_react_agent(llm_with_tools, tools=tools)

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
messages = [HumanMessage(content="What is the temperature in New York and explain me how the stock exchange works?")]
messages = pre_built_agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
