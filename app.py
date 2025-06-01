import streamlit as st
from chatbot import Chatbot

st.set_page_config(page_title="LLaMA3 RAG Chatbot", layout="wide")
st.title("🤖 LLaMA3 Chatbot with PDF RAG")

pdf_path = "1301204142___TA__Atalla_Final.pdf"
bot = Chatbot(pdf_path)

user_input = st.text_input("Ask a question about the uploaded research paper:")

if st.button("Ask") and user_input:
    with st.spinner("LLaMA3 is thinking..."):
        result = bot.ask(user_input)

        st.markdown("### 🧠 Question:")
        st.write(result['question'])

        st.markdown("### 📚 Retrieved Context:")
        st.info(result['context'])

        st.markdown("### 🤖 LLaMA3 Answer:")
        st.success(result['answer'])
