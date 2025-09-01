# Utils
from utils import ollama_check_model, ollama_generator
# App and models
import streamlit as st

# ----------------------------- MAIN -----------------------------

def main():
    
    st.set_page_config(
        page_title="Chatea con Ollama",
        page_icon="🦜",
    )
    st.title("🦜 Chatea con Ollama")

    # Verificar si hay modelos cargados en Ollama
    ollama_check_model("http://ollama:11434", "Ollama")

    # Display chat messages from history on app rerun
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create the chat interface
    if query := st.chat_input("Enter your query here..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": query})

        # Get response from the model
        with st.chat_message('assistant'):
            response = st.write_stream(ollama_generator(st.session_state.selected_model, st.session_state.chat_messages))

        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
