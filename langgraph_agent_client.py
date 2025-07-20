from langgraph_sdk import get_client
import asyncio

client = get_client(url="http://localhost:2024")

async def main():
    async for chunk in client.runs.stream(
        None,  # Threadless run
        "agent", # Name of assistant. Defined in langgraph.json.
        input={
        "messages": [{
            "role": "human",
            "content":"what is weather like in New York",
            }],
        },
    ):
        # print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data['messages'])
        print("\n\n")

asyncio.run(main())