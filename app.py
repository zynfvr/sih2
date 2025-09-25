from flask import Flask, render_template_string, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import traceback
import json

# Import your existing LLM chat functionality
from llm_chat import hybrid_answer, get_db_connection  # Import connection function instead of con

app = Flask(__name__)
CORS(app)

# Store the HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Argo Float Data Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .chat-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            width: 90%;
            max-width: 800px;
            height: 700px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(45deg, #2c3e50, #3498db);
            color: white;
            padding: 20px;
            text-align: center;
            position: relative;
        }
        
        .chat-header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }
        
        .chat-header p {
            opacity: 0.9;
            font-size: 14px;
        }
        
        .status-indicator {
            position: absolute;
            top: 20px;
            right: 20px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #27ae60;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            scroll-behavior: smooth;
        }
        
        .message {
            margin-bottom: 20px;
            display: flex;
            align-items: flex-start;
            animation: slideIn 0.3s ease-out;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            flex-direction: row-reverse;
        }
        
        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: white;
            flex-shrink: 0;
        }
        
        .user .message-avatar {
            background: #3498db;
            margin-left: 10px;
        }
        
        .assistant .message-avatar {
            background: #2c3e50;
            margin-right: 10px;
        }
        
        .message-content {
            background: white;
            padding: 15px 20px;
            border-radius: 20px;
            max-width: 70%;
            word-wrap: break-word;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            position: relative;
        }
        
        .user .message-content {
            background: #3498db;
            color: white;
        }
        
        .assistant .message-content {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
        }
        
        .sql-block {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            overflow-x: auto;
            white-space: pre-wrap;
        }
        
        .result-table {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            font-size: 13px;
            overflow-x: auto;
            white-space: pre-wrap;
        }
        
        .chat-input-container {
            padding: 20px;
            background: #f8f9fa;
            border-top: 1px solid #e9ecef;
        }
        
        .chat-input-form {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .chat-input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e9ecef;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: all 0.3s ease;
        }
        
        .chat-input:focus {
            border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }
        
        .send-button {
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white;
            border: none;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        }
        
        .send-button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
        }
        
        .send-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading-dots {
            display: inline-flex;
            gap: 2px;
        }
        
        .loading-dots span {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: white;
            animation: bounce 1.4s ease-in-out infinite both;
        }
        
        .loading-dots span:nth-child(1) { animation-delay: -0.32s; }
        .loading-dots span:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0.8); }
            40% { transform: scale(1); }
        }
        
        .welcome-message {
            text-align: center;
            color: #666;
            padding: 40px 20px;
            border: 2px dashed #ddd;
            border-radius: 15px;
            margin: 20px;
        }
        
        .welcome-message h3 {
            margin-bottom: 10px;
            color: #2c3e50;
        }
        
        .example-questions {
            margin-top: 20px;
            text-align: left;
        }
        
        .example-questions h4 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .example-questions ul {
            list-style: none;
            padding: 0;
        }
        
        .example-questions li {
            background: #f8f9fa;
            margin: 5px 0;
            padding: 10px 15px;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 1px solid #e9ecef;
        }
        
        .example-questions li:hover {
            background: #e9ecef;
            transform: translateX(5px);
        }
        
        .error-message {
            background: #ffe6e6;
            color: #d32f2f;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #d32f2f;
        }
        
        @media (max-width: 768px) {
            .chat-container {
                width: 95%;
                height: 90vh;
                border-radius: 15px;
            }
            
            .chat-header h1 {
                font-size: 20px;
            }
            
            .message-content {
                max-width: 85%;
            }
            
            .chat-input {
                font-size: 16px;
            }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <div class="status-indicator"></div>
            <h1>🌊 Argo Float Data Assistant</h1>
            <p>Ask questions about oceanographic data and get intelligent insights</p>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="welcome-message">
                <h3>Welcome to your Argo Float Data Assistant! 🤖</h3>
                <p>I can help you explore oceanographic data using both SQL queries and intelligent document retrieval. Ask me anything about your Argo float data!</p>
                
                <div class="example-questions">
                    <h4>Try asking:</h4>
                    <ul>
                        <li onclick="sendExampleQuestion(this)">What are the maximum temperatures recorded?</li>
                        <li onclick="sendExampleQuestion(this)">Show me data for float 1901393</li>
                        <li onclick="sendExampleQuestion(this)">How many unique floats are in the dataset?</li>
                        <li onclick="sendExampleQuestion(this)">What's the deepest measurement recorded?</li>
                        <li onclick="sendExampleQuestion(this)">Tell me about temperature variations</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="chat-input-container">
            <form class="chat-input-form" id="chatForm">
                <input 
                    type="text" 
                    class="chat-input" 
                    id="chatInput" 
                    placeholder="Ask about your Argo float data..."
                    autocomplete="off"
                />
                <button type="submit" class="send-button" id="sendButton">
                    <span>➤</span>
                </button>
            </form>
        </div>
    </div>

    <script>
        class ArgoChatbot {
            constructor() {
                this.chatMessages = document.getElementById('chatMessages');
                this.chatForm = document.getElementById('chatForm');
                this.chatInput = document.getElementById('chatInput');
                this.sendButton = document.getElementById('sendButton');
                
                this.initializeEventListeners();
            }
            
            initializeEventListeners() {
                this.chatForm.addEventListener('submit', (e) => {
                    e.preventDefault();
                    this.sendMessage();
                });
                
                this.chatInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        this.sendMessage();
                    }
                });
            }
            
            async sendMessage() {
                const message = this.chatInput.value.trim();
                if (!message) return;
                
                this.chatInput.value = '';
                this.setLoading(true);
                this.addMessage(message, 'user');
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ message: message })
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        this.addMessage(`Error: ${data.error}`, 'assistant', true);
                    } else {
                        this.addMessage(data.response, 'assistant');
                    }
                } catch (error) {
                    this.addMessage('Sorry, I encountered an error processing your request. Please try again.', 'assistant', true);
                    console.error('Error:', error);
                } finally {
                    this.setLoading(false);
                }
            }
            
            addMessage(content, sender, isError = false) {
                const welcomeMsg = this.chatMessages.querySelector('.welcome-message');
                if (welcomeMsg) {
                    welcomeMsg.remove();
                }
                
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}`;
                
                const avatar = document.createElement('div');
                avatar.className = 'message-avatar';
                avatar.textContent = sender === 'user' ? 'U' : '🤖';
                
                const messageContent = document.createElement('div');
                messageContent.className = 'message-content';
                
                if (isError) {
                    messageContent.innerHTML = `<div class="error-message">${content}</div>`;
                } else {
                    messageContent.innerHTML = this.formatMessageContent(content);
                }
                
                messageDiv.appendChild(avatar);
                messageDiv.appendChild(messageContent);
                
                this.chatMessages.appendChild(messageDiv);
                this.scrollToBottom();
            }
            
            formatMessageContent(content) {
                // Extract SQL blocks (using proper JavaScript regex)
                content = content.replace(/```sql\\n([\\s\\S]*?)\\n```/g, '<div class="sql-block">$1</div>');
                content = content.replace(/✅ SQL executed:\\n([\\s\\S]*?)\\n\\n/g, '✅ SQL executed:\\n<div class="sql-block">$1</div>\\n\\n');
                
                // Extract result blocks
                content = content.replace(/📊 Result:\\n([\\s\\S]*?)(?=\\n\\n|$)/g, '📊 Result:\\n<div class="result-table">$1</div>');
                
                // Convert line breaks to <br>
                content = content.replace(/\\n/g, '<br>');
                
                return content;
            }
            
            setLoading(isLoading) {
                if (isLoading) {
                    this.sendButton.disabled = true;
                    this.sendButton.innerHTML = '<div class="loading-dots"><span></span><span></span><span></span></div>';
                    this.chatInput.disabled = true;
                } else {
                    this.sendButton.disabled = false;
                    this.sendButton.innerHTML = '<span>➤</span>';
                    this.chatInput.disabled = false;
                    this.chatInput.focus();
                }
            }
            
            scrollToBottom() {
                this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
            }
        }
        
        function sendExampleQuestion(element) {
            const question = element.textContent;
            document.getElementById('chatInput').value = question;
            chatbot.sendMessage();
        }
        
        const chatbot = new ArgoChatbot();
        
        window.addEventListener('load', () => {
            document.getElementById('chatInput').focus();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Serve the main chat interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages and return responses"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        print(f"User question: {user_message}")  # Debug logging
        
        # Use your existing hybrid_answer function
        response = hybrid_answer(user_message)
        
        print(f"LLM response: {response}")  # Debug logging
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        con = get_db_connection()
        test_query = "SELECT COUNT(*) as count FROM floats LIMIT 1"
        result = con.execute(test_query).fetchone()
        con.close()
        
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'floats_available': result[0] if result else 0
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/stats')
def get_stats():
    """Get basic statistics about the dataset"""
    try:
        con = get_db_connection()
        stats = {}
        
        # Get float count
        result = con.execute("SELECT COUNT(DISTINCT platform_number) as float_count FROM floats").fetchone()
        stats['total_floats'] = result[0] if result else 0
        
        # Get measurement count
        result = con.execute("SELECT COUNT(*) as measurement_count FROM measurements").fetchone()
        stats['total_measurements'] = result[0] if result else 0
        
        # Get cycle count
        result = con.execute("SELECT COUNT(*) as cycle_count FROM cycles").fetchone()
        stats['total_cycles'] = result[0] if result else 0
        
        con.close()
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Argo Float Chatbot Server...")
    print("📊 Testing database connection...")
    
    try:
        # Test your existing setup
        con = get_db_connection()
        test_result = con.execute("SELECT COUNT(*) FROM floats LIMIT 1").fetchone()
        print(f"✅ Database connected! Found floats table with {test_result[0]} records")
        con.close()
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)
    
    print("🌊 Argo Float Data Assistant ready!")
    print("🔗 Open http://localhost:5000 in your browser")
    
    # Run Flask app
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )