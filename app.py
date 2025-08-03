import streamlit as st
import sys
import re
import time
import os
import tempfile
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from templates import css, bot_template, user_template

# Load environment variables
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.error("Missing API key. Please set OPENAI_API_KEY in your .env file.")
    st.stop()

st.set_page_config(page_title='Chat with multiple PDFs', page_icon=':books:')

# ==============================
# OCR Extraction with Temp Files (Optimized for Math)
# ==============================
def extract_text_with_ocr(pdf_files):
    text = ""
    for uploaded_file in pdf_files:
        uploaded_file.seek(0)  # Reset pointer
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        images = convert_from_path(tmp_path, dpi=300)
        for img in images:
            text += pytesseract.image_to_string(img, lang="eng", config="--oem 1 --psm 6") + "\n"

        os.remove(tmp_path)
    return text

# ==============================
# Smart PDF Text Extraction with OCR fallback
# ==============================
def get_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

    # OCR fallback if text seems incomplete
    if (
        len(text.strip()) < 100 or
        "____" in text or
        ("Question" in text and "=" not in text)
    ):
        st.warning("Text seems incomplete. Switching to OCR...")
        text = extract_text_with_ocr(pdf_docs)

    return text

# ==============================
# Text Chunking (Reduced size for better accuracy)
# ==============================
def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\nProblem", "\nQuestion", "\n\n", "\n", ".", " ", ""],
        chunk_size=1200,  # Reduced for tighter context
        chunk_overlap=150
    )
    return splitter.split_text(text)

# ==============================
# FAISS Vector Store
# ==============================
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# ==============================
# Translation (User Query → English)
# ==============================

def translate_to_english(text):
    # Detect if text is mostly English letters and allowed math symbols
    if re.fullmatch(r"[A-Za-z0-9\s\-\+\=\(\)\[\]\{\}\.,:;!?/*^∞Σ∫→√π]+", text):
        return text.strip()  # Skip translation for English input

    # If non-English characters found, call GPT for translation
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    prompt = f"Translate the following text to English without adding explanations. Keep all math symbols unchanged:\n\n{text}\n\nReturn only the translated text."
    return llm.predict(prompt).strip()

# ==============================
# Conversation Chain with Updated Prompt
# ==============================
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)

    # Plain text with math symbols
    system_prompt = """
    You are an expert math tutor. Solve the question clearly using normal math symbols in plain text (not LaTeX).

    Rules:
    - DO NOT use LaTeX or MathJax ($ or $$).
    - Use real math symbols where possible:
      ∫ for integrals, Σ for summation, √ for square root, ∞ for infinity, → for limits.
    - Write functions like f(x), g(x).
    - For exponents, use superscript (x²) or x^2 if needed.
    - Show all steps like a textbook, each on a new line.
    - Include the original question before solving.
    - At the end, write "Conclusion: ..." with the final answer.

    Example:
    Solution:
    1. Question: Solve 3x - 7 = 14.
    2. Add 7 to both sides:
       3x = 21
    3. Divide by 3:
       x = 7
    Conclusion: x = 7
    """

    messages = [
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("Context: {context}\n\nQuestion: {question}")
    ]

    prompt = ChatPromptTemplate.from_messages(messages)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # Increase k for better retrieval
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# ==============================
# Display Chat
# ==============================
def handle_userinput(user_question):
    translated_question = translate_to_english(user_question)
    response = st.session_state.conversation({'question': translated_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# ==============================
# Main App
# ==============================
def main():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('Chat with multiple PDFs :books:')
    user_question = st.text_input("Ask questions in ANY language about your documents")
    if user_question:
        if st.session_state.conversation is None:
            st.warning("Please upload and process your PDFs first!")
        else:
            handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click 'Process'", accept_multiple_files=True)
        process_button = st.button("Process")

        if process_button:
            if not pdf_docs:
                st.error("Please upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No text found. Make sure the PDFs have text or use OCR.")
                        return
                    text_chunks = get_chunks(raw_text)
                    vector_store = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vector_store)
                    st.success("PDFs processed successfully!")

if __name__ == '__main__':
    main()
