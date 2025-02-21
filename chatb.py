import openai
from dotenv import load_dotenv
import os
import json
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

# Load environment variables securely
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check API key
if not openai_api_key:
    raise ValueError("OpenAI API Key is missing! Set it in the .env file.")

# Load the syllabus from a JSON file
def load_syllabus(file_path):
    """Load syllabus topics from a JSON file. Handle missing file error."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The syllabus file '{file_path}' was not found. Please create it.")

    with open(file_path, "r") as f:
        return json.load(f)

# Define syllabus file path
syllabus_file = "syllabus.json"
syllabus = load_syllabus(syllabus_file)

# Track chatbot conversation context
conversation_context = {"last_bot_message": None}  # Stores last message from chatbot

# Extract syllabus topics and subtopics
def get_allowed_topics():
    """Extract syllabus topics and subtopics."""
    allowed_topics = list(syllabus.keys())  # Main topics
    allowed_subtopics = [subtopic for subtopics in syllabus.values() for subtopic in subtopics]  # Subtopics
    return allowed_topics + allowed_subtopics  # Combine both

# Get OpenAI embeddings for a given text
def get_embedding(text):
    """Get vector embeddings using OpenAI API (Updated for openai>=1.0.0)."""
    client = openai.OpenAI(api_key=openai_api_key)  # New API client
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding)

# Precompute embeddings for syllabus topics
syllabus_embeddings = {topic: get_embedding(topic) for topic in get_allowed_topics()}

# Check similarity between query and syllabus topics using AI embeddings
def is_query_allowed(query):
    """Determine if a user's query is related to the syllabus using embeddings."""
    query_lower = query.lower()

    # Allow short responses for conversation flow
    if is_general_response(query_lower):
        return True  

    # Allow response if chatbot recently asked a follow-up question
    if conversation_context["last_bot_message"]:
        return True  

    # Compute similarity between query and syllabus embeddings
    query_embedding = get_embedding(query)
    similarities = {
        topic: np.dot(query_embedding, topic_embedding) / 
               (np.linalg.norm(query_embedding) * np.linalg.norm(topic_embedding))
        for topic, topic_embedding in syllabus_embeddings.items()
    }

    # Find best match
    best_match = max(similarities, key=similarities.get)
    best_similarity = similarities[best_match]

    print(f"🔍 Query: {query} | Best Match: {best_match} | Similarity Score: {best_similarity}")

    return best_similarity > 0.75  # Accept query if similarity score is high

# Define general responses like "yes", "no", etc.
def is_general_response(query):
    """Check if the query is a short general response."""
    general_responses = ["yes", "no", "maybe", "okay", "sure", "thanks", "thank you"]
    return query.lower() in general_responses

# Define the ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

# System Prompt with syllabus restriction and interactive learning
system_prompt = """
You are a Web Development Tutor. Your task is to guide the student **only within the syllabus topics**.
You must also encourage interaction and conversation.

If a user asks a question, provide an explanation and **ask them a follow-up question**.
If they reply with something unrelated, try to keep the conversation flowing.

If a user gives a short response like "Yes", "Okay", or "Sure", acknowledge it and move forward.

### **Example Responses**
User: "How do I make a table in HTML?"
Bot: "Tables in HTML are created using the <table> tag. Would you like an example of a basic table?"

User: "Yes"
Bot: "Great! Here’s an example of an HTML table..."
"""

# Convert syllabus dictionary to formatted string
syllabus_topics_str = "\n".join(
    [f"- {topic}: {', '.join(subtopics)}" for topic, subtopics in syllabus.items()]
)

# Format the prompt with the syllabus topics
formatted_prompt = system_prompt.format(syllabus_topics=syllabus_topics_str)

# Define the chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([("system", formatted_prompt), ("human", "{question}")])

# Chain the prompt with the model
chain = chat_prompt | model | StrOutputParser()

# Function to handle chatbot response
def get_chat_response(user_question):
    """Allow free conversation but restrict non-syllabus topics using AI-based query detection."""

    if is_general_response(user_question):
        return "Got it! What would you like to learn next in web development?"

    if is_query_allowed(user_question):
        response = chain.invoke({"question": user_question})  # Allow free conversation within syllabus topics
        conversation_context["last_bot_message"] = response  # Store chatbot's last response
        return response

    return "I'm here to help with web development topics. Could you ask something related to HTML, CSS, or frontend development?"

#-- Chat loop
#print("Chatbot is running! Type 'bye' or 'exit' to stop.")
#while True:
#    try:
#        user_input = input("You: ").strip()
#        if user_input.lower() in ["bye", "exit"]:
#            print("Chatbot: Goodbye! Have a great day!")
#            break
#
#        response = get_chat_response(user_input)
#        print(f"Chatbot: {response}")
#
#   except Exception as e:
#       print(f"Chatbot: Oops! Something went wrong. Error: {e}")
