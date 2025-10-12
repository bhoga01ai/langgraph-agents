# LangGraph Agent and Workflow Examples

This repository provides a collection of Python scripts demonstrating various features of the LangGraph library. These examples showcase how to build tool-using agents, create prompt-chaining workflows, and interact with deployed LangGraph agents.

## Getting Started

Follow these steps to set up and run the examples:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/week2-langgraph-agents.git
    cd week2-langgraph-agents
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    uv venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    uv pip install -r requirements.txt
    ```

4.  **Set up environment variables**:
    Create a `.env` file by copying the `env.example` file:
    ```bash
    cp env.example .env
    ```
    Then, add your API keys to the `.env` file.

## Table of Contents

- [Overview](#overview)
- [Features](#features)

- [Usage](#usage)
  - [Tool-Using Agent](#tool-using-agent)
  - [Prompt-Chaining Workflow](#prompt-chaining-workflow)
  - [Agent Client](#agent-client)
  - [Pinecone Retriever](#pinecone-retriever)
- [Dependencies](#dependencies)

## Overview

This project serves as a practical guide to using LangGraph for building sophisticated language agent applications. It includes four distinct examples:

1.  **Tool-Using Agent (`4_langgraph_tools_agent.py`)**: Demonstrates how to create an agent that can use a variety of tools, including fetching weather information, currency exchange rates, stock prices, and searching YouTube.
2.  **Prompt-Chaining Workflow (`2_langgraph_prompt_chain_workflow.py`)**: Illustrates a conditional workflow where a joke is generated, evaluated, and then improved based on a quality check.
3.  **Agent Client (`langgraph_agent_client.py`)**: Provides a client script to communicate with a deployed LangGraph agent.
4.  **Pinecone Retriever (`pinecone_retriever_tool.py`)**: Shows how to create a tool that retrieves information from a Pinecone vector store.

## Features

-   **Tool Integration**: Shows how to integrate external tools into a LangGraph agent.
-   **Conditional Workflows**: Demonstrates the use of conditional edges to create dynamic and responsive agent behaviors.
-   **State Management**: Utilizes `StateGraph` and `MessagesState` to manage the application's state throughout the workflow.
-   **Pre-built Agents**: Leverages `create_react_agent` for rapid development of tool-using agents.
-   **Visualization**: Generates and saves visual representations of the workflows as PNG images and Mermaid diagram files.
-   **Vector Store Integration**: Shows how to connect to a Pinecone vector store and retrieve data.
-   **Utility Functions**: The `util.py` file provides helper functions for creating and managing the Pinecone index, including text splitting, embedding generation, and Pinecone index management.



## Usage

### Tool-Using Agent

This script demonstrates how to build an agent that can use multiple tools.

**To run the script**:
```bash
python 4_langgraph_tools_agent.py
```

This will execute a pre-defined query ("What is the temperature in New York and explain me how the stock exchange works?") and print the agent's response.

### Prompt-Chaining Workflow

This script showcases a workflow for generating and improving a joke based on a conditional check.

**To run the script**:
```bash
python 2_langgraph_prompt_chain_workflow.py
```

The script will generate a joke about "cats," check if it has a punchline, and then improve and polish it if necessary. The final joke will be printed to the console.

### langgraph studio using langgraph dev

**To run the script**:
1. activate the virtual environment
2. run the following command:
```bash
langgraph dev
```
This will start the langgraph studio.

### Agent Client

This script provides a client to interact with a deployed LangGraph agent.

**To run the script**:
1.  First, ensure your LangGraph agent is running and accessible at `http://localhost:2024`.
2.  Then, run the client script:
    ```bash
    python langgraph_agent_client.py
    ```

The client will send a request to the agent ("what is weather like in New York") and print the streamed response.

### Pinecone Retriever

This script demonstrates how to create a tool that retrieves information from a Pinecone vector store. The `pinecone_retriever_tool.py` file contains the tool definition, and the `util.py` file contains helper functions for creating and managing the Pinecone index. The `docs/apple_10k.pdf` file is used as the data source for the vector store.

**To run the script**:
```bash
python pinecone_retriever_tool.py
```

This will execute a pre-defined query ("what was obama said about schools") and print the retrieved context.

### Chatbot Example

This script (`1_chatbot.py`) implements a simple conversational chatbot using LangGraph. It demonstrates how to create a basic agent that can respond to user input. The `chatbot_graph.png` file provides a visual representation of the chatbot's workflow.

**To run the script**:
```bash
python 1_chatbot.py
```

This will start an interactive chatbot session in your terminal.



## Dependencies

The project's dependencies are listed in the `requirements.txt` file:

- `yfinance`
- `python-dotenv`
- `google-genai`
- `langchain`
- `langchain-google-genai`
- `langchain-core`
- `langchain-community`
- `youtube_search`
- `langgraph`
- `IPython`
- `langgraph-cli[inmem]`
- `langgraph-sdk`
- `langchain-pinecone`
- `langchain-openai`
- `pinecone`
- `langchain-huggingface`
- `pypdf`
- `sentence-transformers`