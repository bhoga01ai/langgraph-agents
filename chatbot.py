##               Chatbot using LangGraph
##  compoents
##   1. LLM  -- Gemini 2.5 with API key
##   2. Memory  -- Memory to save the conversation history within the Chat of same session for recall purposes-longterm memory
##   3. Graph  -- LangGraph is a workflow graph where each node is a component and connected
##   4 .State  -- It's scrachPad where we store the conversation histrory b/w the nodes(componets) of workflow
##                (/tmp memoery - shortterm)

# STEP 1 API keys
from dotenv import load_dotenv
load_dotenv()

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
from IPython.display import display,Image
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode


# STEP 2 Define the LLM
llm = init_chat_model(model="gemini-2.5-flash",model_provider="google_genai",temperature=0.5)

# STEP 3 Defin the Chatbot STATE -- ScrachPAD

class State(TypedDict):
    messages: Annotated[list,add_messages]

# STEP 4 Define the CHATBOT node

def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# STEP 5 Define the graph
graph_builder = StateGraph(State)

# STEP 6 Add the nodes
graph_builder.add_node('chatbot',chatbot)
graph_builder.add_edge(START,'chatbot')
graph_builder.add_edge('chatbot',END)

# STEP 7 compile, save and then  run
memory_saver=InMemorySaver()

graph_without_memory=graph_builder.compile()   # without mememory

# save the graph
png_data = graph_without_memory.get_graph().draw_mermaid_png()
with open("chatbot_graph.png", "wb") as f:
    f.write(png_data)
print("Graph saved as chatbot_graph.png")
display(Image(filename='chatbot_graph.png'))


# STEP 8 Invoke the graph without memory
print("------------------------")
messages=[
    {"role":"user","content":"why sky is blue?"}
]

response=graph_without_memory.invoke({"messages":messages})
print(response['messages'][-1].content)

print("------------------------")
messages=[
    {"role":"user","content":"what was previous question?"}
]

response=graph_without_memory.invoke({"messages":messages})
print(response['messages'][-1].content)

# STEP 9 Invoke the graph with memory
graph_with_memory=graph_builder.compile(checkpointer=memory_saver) 
config = {"configurable": {"thread_id": "1"}}

# The config is the **second positional argument** to stream() or invoke()!
print("------------------------")
while True:
    user_input=input("You: ")
    if user_input.lower() in ["exit","quit"]:
        break
    events = graph_with_memory.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()