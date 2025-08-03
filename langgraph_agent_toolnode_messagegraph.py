## Langgraph Toolnode agent
# Define Tools: Create custom Python functions to act as tools your AI application can use.
# Define Router: Set up routing logic to control conversation flow and tool selection.
# Build a LangGraph Application: Structure your application using LangGraph, including the Gemini model and custom tools that you define.
# Local Testing: Test your LangGraph application locally to ensure functionality.


from typing import Literal
from urllib import response
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph, START
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
import os
from IPython.display import Image, display

# STEP 1 : LOAD ENV variables
from dotenv import load_dotenv
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# STEP 2 : Define the LLM
llm = init_chat_model(model="gemini-2.5-flash",model_provider="google_genai",temperature=0.5)


# STEP 3 : Define the ToolNODE Function
def get_product_details(product_name: str):
    """Gathers basic details about a product."""
    details = {
        "smartphone": "A cutting-edge smartphone with advanced camera features and lightning-fast processing.",
        "coffee": "A rich, aromatic blend of ethically sourced coffee beans.",
        "shoes": "High-performance running shoes designed for comfort, support, and speed.",
        "headphones": "Wireless headphones with advanced noise cancellation technology for immersive audio.",
        "speaker": "A voice-controlled smart speaker that plays music, sets alarms, and controls smart home devices.",
    }
    return details.get(product_name, "Product details not found.")

tool_node = ToolNode([get_product_details])

llm_with_tools = llm.bind_tools([get_product_details])

# STEP 4 : Define the router
def router(state: list[BaseMessage]) -> Literal["get_product_details", "__end__"]:
    """Initiates product details retrieval if the user asks for a product."""
    # Get the tool_calls from the last message in the conversation history.
    tool_calls = state[-1].tool_calls
    # If there are any tool_calls
    if len(tool_calls):
        print("tool_name:",tool_calls[0]['name'])
        # Return the name of the tool to be called
        return "get_product_details"
    else:
        # End the conversation flow.
        return END  # Use END instead of "__end__"

# STEP 5 : Define workflow 
builder = MessageGraph()
builder.add_node("llm_with_tools", llm_with_tools)
builder.add_node("get_product_details", tool_node)
builder.add_edge(START, "llm_with_tools")
# Use conditional_edges instead of add_edge for routing
builder.add_conditional_edges("llm_with_tools", router)
builder.add_edge("get_product_details", END)

graph = builder.compile()

graph_data = graph.get_graph().draw_mermaid_png()
with open("langgraph_toolnode_messagegraph.png", "wb") as f:  # Use "wb" for binary
    f.write(graph_data)
print("Graph saved as langgraph_toolnode_messagegraph.png")
display(Image(filename='langgraph_toolnode_messagegraph.png'))


# The config is the **second positional argument** to stream() or invoke()!
print("------------------------")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    messages = [HumanMessage(user_input)]  # Pass as list
    response = graph.invoke(messages)
    print(response[-1].content)  # response is a list of messages