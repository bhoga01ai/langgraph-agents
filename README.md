# LangGraph Agents

This project demonstrates various examples of building AI agents and workflows using the LangGraph library. It includes a simple sequential workflow, a tool-using agent, and a ReAct agent with a comprehensive set of tools and a Flask UI.

## Features

- **Sequential Workflow**: A simple workflow for generating and improving jokes.
- **Tool-Using Agent**: An agent that can use a tool to get product details.
- **ReAct Agent**: A more advanced agent that can use a variety of tools, including:
    - Weather lookup
    - Currency exchange rates
    - Stock prices
    - YouTube search
    - Pinecone vector store for RAG (Retrieval-Augmented Generation)
- **Flask UI**: A simple web interface to interact with the ReAct agent.

## Architecture

The project is structured as follows:

- **`langgraph_prompt_chain_workflow.py`**: Implements the sequential joke generation workflow.
- **`langgraph_agent_toolnode_messagegraph.py`**: Implements the simple tool-using agent.
- **`langgraph_tools_agent.py`**: Implements the ReAct agent with multiple tools.
- **`flask_ui.py`**: Provides a Flask-based web UI for the agent.
- **`util.py`**: Contains utility functions for working with Pinecone.

## Getting Started

### Prerequisites

- Python 3.10+
- An environment with the required packages installed.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/week2-langgraph-agents.git
    cd week2-langgraph-agents
    ```
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Set up your environment variables by creating a `.env` file from the `env.example`:
    ```bash
    cp env.example .env
    ```
    Then, fill in your API keys in the `.env` file.

### Running the Agent

1.  Start the LangGraph agent:
    ```bash
    langgraph up
    ```
2.  Start the Flask UI:
    ```bash
    python flask_ui.py
    ```
3.  Open your browser and navigate to `http://localhost:5000` to interact with the agent.

## Tools

The ReAct agent is equipped with the following tools:

- `get_temperature`: Get the current temperature for a city.
- `get_currency_exchange_rates`: Get currency exchange rates.
- `get_stock_price`: Get the latest stock price for a ticker.
- `youtube_search`: Search for videos on YouTube.
- `retrieve_obama_speech_context`: Retrieve context from an Obama speech vector store.
- `ingest_apple_10k_docs_into_vector_store`: Ingest Apple's 10-K report into a vector store.
- `retrieve_apple_10k_context`: Retrieve context from the Apple 10-K vector store.
- `drop_pinecone_index`: Delete a Pinecone index.
- `list_pinecone_indexes`: List all available Pinecone indexes.
- `get_pinecone_index_details`: Get detailed information about a Pinecone index.
- `handle_file_upload`: Handle file uploads from the LangStudio interface.
- `helper_func`: Get information about all available tools.
