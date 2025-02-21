from flask import Flask, render_template, request, jsonify
from chatb import get_chat_response  # Import chatbot function

app = Flask(__name__)

@app.route("/")
def index():
    """Serve the chatbot UI."""
    return render_template("index.html")  # Load web UI

@app.route("/get", methods=["POST"])
def chat():
    """Handles chatbot responses."""
    user_message = request.json["msg"]  # Get user input
    bot_response = get_chat_response(user_message)  # Get chatbot response
    return jsonify({"response": bot_response})  # Send response back

if __name__ == "__main__":
    print("\n🌐 Flask Web UI running at: http://127.0.0.1:5000/")
    app.run(debug=True, use_reloader=False)  # Start Flask
