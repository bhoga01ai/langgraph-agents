#   LangGraph  Prompt Chain Workflow - Sequential flow

# Components
# 1. Define a lLM 
# 2. Define a State  --  A State is a variable that can be used to store the output of a previous component
# 3  Defining the nodes  -  A node is a component that can be executed or a default node - START and END or  A Python fucntion 
# 4. Define a Workflow  --  A Workflow is a series of components that are executed in order
#       Workflow is made of Nodes and Edges
#       Nodes are the components  -  A function that is executed or default nodes - START and END
#       Edges are the connections between nodes  - It is also a python funtion that return the node and it gives direction of the flow
#            Two types of Edges  - 1.  Normal Edges  2. Conditional edges


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
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

# STEP 1: Define the LLM
llm = init_chat_model(model="gemini-2.5-flash",model_provider="google_genai",temperature=0.5)
# a Simple test 
# response=llm.invoke("who are you")
# print(response.content)

# STEP 2: Define the State
class State(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str

# STEP 3. Define Nodes
def generate_joke(state: State):
    """First LLM call to generate initial joke"""

    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}

def check_punchline(state: State):
    """Gate function to check if the joke has a punchline"""

    # Simple check - does the joke contain "?" or "!"
    if "?" in state["joke"] or "!" in state["joke"]:
        return "Pass"
    return "Fail"

def improve_joke(state: State):
    """Second LLM call to improve the joke"""

    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}


def polish_joke(state: State):
    """Third LLM call for final polish"""

    msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
    return {"final_joke": msg.content}

# STEP 4. CREATE A workflow to be executed

# Build workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)
workflow.add_node("polish_joke", polish_joke)

# Add edges to connect nodes
workflow.add_edge(START, "generate_joke")
workflow.add_conditional_edges(
    "generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END}
)
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)

# Compile
graph = workflow.compile()

# Show workflow and save it

# To save as a PNG image file:
png_data = graph.get_graph().draw_mermaid_png()
with open("workflow_graph.png", "wb") as f:
    f.write(png_data)
print("Graph saved as workflow_graph.png")

# To save as a Mermaid diagram file (.mmd):
mermaid_text = graph.get_graph().draw_mermaid()
with open("workflow_graph.mmd", "w") as f:
    f.write(mermaid_text)
print("Graph saved as workflow_graph.mmd")

# To display it in a notebook, you can still use:
display(Image(png_data))

# Invoke
state = graph.invoke({"topic": "cats"})
print("Initial joke:")
print(state["joke"])
print("\n--- --- ---\n")
if "improved_joke" in state:
    print("Improved joke:")
    print(state["improved_joke"])
    print("\n--- --- ---\n")

    print("Final joke:")
    print(state["final_joke"])
else:
    print("Joke failed quality gate - no punchline detected!")
