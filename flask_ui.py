from flask import Flask, render_template, request, jsonify, Response
import requests
import json
import asyncio
from langgraph_sdk import get_client

app = Flask(__name__)

# LangGraph client
client = get_client(url="http://localhost:2024")

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>LangGraph Agent UI</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .chat-container { border: 1px solid #ddd; height: 400px; overflow-y: scroll; padding: 10px; margin-bottom: 10px; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user { background-color: #e3f2fd; text-align: right; }
        .ai { background-color: #f5f5f5; }
        .input-container { display: flex; }
        #messageInput { flex: 1; padding: 10px; border: 1px solid #ddd; }
        #sendButton { padding: 10px 20px; background-color: #2196f3; color: white; border: none; cursor: pointer; }
    </style>
</head>
<body>
    <h1>LangGraph Agent Chat</h1>
    <div id="chatContainer" class="chat-container"></div>
    <div class="input-container">
        <input type="text" id="messageInput" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
        <button id="sendButton" onclick="sendMessage()">Send</button>
    </div>

    <script>
        function addMessage(content, isUser) {
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + (isUser ? 'user' : 'ai');
            messageDiv.textContent = content;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message) return;

            addMessage(message, true);
            input.value = '';

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });

                const data = await response.json();
                addMessage(data.response, false);
            } catch (error) {
                addMessage('Error: ' + error.message, false);
            }
        }

        // Add welcome message
        addMessage('Hello! I\'m your LangGraph agent. How can I help you today?', false);
    </script>
</body>
</html>
    '''

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data['message']
        
        # Use asyncio to run the async client
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def get_response():
            final_ai_content = None
            
            async for chunk in client.runs.stream(
                None,
                "agent",
                input={
                    "messages": [{
                        "role": "human",
                        "content": message,
                    }],
                },
            ):
                if chunk.event == 'values' and hasattr(chunk, 'data') and 'messages' in chunk.data:
                    messages = chunk.data['messages']
                    
                    for msg in reversed(messages):
                        if msg.get('type') == 'ai':
                            final_ai_content = msg.get('content', '')
                            break
            
            return final_ai_content or "Sorry, I couldn't process your request."
        
        response = loop.run_until_complete(get_response())
        loop.close()
        
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)