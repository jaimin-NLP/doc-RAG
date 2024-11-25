import streamlit as st
from app.embeddings import EmbeddingsManager
from app.chatbot import ChatbotManager
import os

# Initialize shared state
if "chatbot_manager" not in st.session_state:
    st.session_state["chatbot_manager"] = None
if "temp_pdf_path" not in st.session_state:
    st.session_state["temp_pdf_path"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Streamlit layout
st.set_page_config(page_title="Document Buddy App", layout="wide")

# Sidebar
st.sidebar.title("Document Buddy App")
menu = st.sidebar.radio("Navigation", ["🏠 Home", "📂 Upload & Embed", "🤖 Chat"])

# Home Tab
if menu == "🏠 Home":
    st.title("📄 Welcome to Document Buddy")
    st.write("Upload your documents, create embeddings, and interact with an AI chatbot!")

# Upload & Embed Tab
elif menu == "📂 Upload & Embed":
    st.title("📂 Upload Document & Create Embeddings")

    # File Upload
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_file:
        # Save uploaded file to a temporary location
        temp_path = "temp.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["temp_pdf_path"] = temp_path

        # Display file name and size
        st.success(f"Uploaded: {uploaded_file.name}")
        st.write(f"File size: {uploaded_file.size / 1024:.2f} KB")

    # Create embeddings
    if st.button("🧠 Create Embeddings"):
        if st.session_state["temp_pdf_path"] is None:
            st.warning("Please upload a file first!")
        else:
            try:
                embeddings_manager = EmbeddingsManager(
                    model_name="BAAI/bge-small-en",
                    device="cpu",
                    qdrant_url="http://localhost:6333",
                    collection_name="vector_db",
                )
                with st.spinner("Creating embeddings..."):
                    result = embeddings_manager.create_embeddings(
                        st.session_state["temp_pdf_path"]
                    )
                st.success(result)

                # Initialize ChatbotManager
                st.session_state["chatbot_manager"] = ChatbotManager(
                    model_name="BAAI/bge-small-en",
                    device="cpu",
                    # llm_model="llama3.2:3b",
                    qdrant_url="http://localhost:6333",
                    collection_name="vector_db",
                )
            except Exception as e:
                st.error(f"Error: {e}")

# Chat Tab
elif menu == "🤖 Chat":
    st.title("🤖 Chat with Your Document")
    if st.session_state["chatbot_manager"] is None:
        st.info("Please upload a document and create embeddings first.")
    else:
        # Display chat messages
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

        # User input
        user_input = st.chat_input("Ask your question...")
        if user_input:
            # Append user message immediately to the chat window
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)

            # Placeholder for chatbot response
            response_placeholder = st.chat_message("assistant").empty()

            # Generate and display chatbot response in streaming fashion
            chatbot_response = ""
            try:
                with st.spinner("Generating response..."):
                    for chunk in st.session_state["chatbot_manager"].get_response(user_input):
                        chatbot_response += chunk
                        response_placeholder.write(chatbot_response)
            except Exception as e:
                chatbot_response = f"⚠️ Error generating response: {e}"
                response_placeholder.write(chatbot_response)

            # Append final response to chat history
            st.session_state["messages"].append(
                {"role": "assistant", "content": chatbot_response}
            )