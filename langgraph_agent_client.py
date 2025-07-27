from langgraph_sdk import get_client
import asyncio

client = get_client(url="http://localhost:2024")

async def main():
    final_ai_content = None
    
    async for chunk in client.runs.stream(
        None,  # Threadless run
        "agent", # Name of assistant. Defined in langgraph.json.
        input={
        "messages": [{
            "role": "human",
            "content":"why sky is blue.",
            }],
        },
    ):
        # Extract final AI response
        if chunk.event == 'values' and hasattr(chunk, 'data') and 'messages' in chunk.data:
            messages = chunk.data['messages']
            
            # Find the last AI message
            for message in reversed(messages):
                if message.get('type') == 'ai':
                    final_ai_content = message.get('content', '')
                    break
    
    # Print only the final AI response
    if final_ai_content:
        print("\nFinal AI Response:")
        print(final_ai_content)
    else:
        print("No AI response found")

asyncio.run(main())