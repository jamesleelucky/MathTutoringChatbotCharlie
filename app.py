import streamlit as st
import sys
import re 
import time 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from templates import css, bot_template, user_template
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title='Chat with multiple PDFs', page_icon=':books:')

# ==============================
# PDF Text Extraction
# ==============================
def get_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

# ==============================
# Text Chunking
# ==============================
# def get_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator='\n',
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks
def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\nProblem", "\n\n", "\n", ".", " ", ""],
        chunk_size=20000,  # Larger chunk to preserve context
        chunk_overlap=1200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks

# ==============================
# Vector Store (FAISS)
# ==============================
def get_vectorstore(text_chunks):
    if not text_chunks:  # Prevent IndexError
        return None
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# ==============================
# Conversation Chain with Prompt
# ==============================
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(
        model_name="gpt-4o",  # Or gpt-3.5-turbo if needed
        temperature=0.2
    )

    system_prompt = """
        You are the most intelligent math tutor in this world. Solve ONLY the given question using the provided context.
        Do NOT invent problems that are not in the question.

        Format your response as:

        Solution:
        1. Explain what the equation is. 
            Clearly state the equation using MathJax formatting.
            - For inline math use single dollar signs: $...$
            - For displayed equations use double dollar signs: $$...$$
            DO NOT use square brackets [ ] or raw LaTeX commands without MathJax delimiters.

        2. Show the steps to solve the problem. 
            Clearly use MathJax formatting for all equations.
            - Inline math: $...$
            - Display math: $$...$$
            DO NOT use square brackets [ ] for equations.

        3. Conclusion:
            State the final result.

        Rules:
        - Use clear English and MathJax for math.
        - Do NOT approximate unless required.
        - DO NOT use raw LaTeX outside of MathJax delimiters.
        If possible, compute the numeric value.
    """


    messages = [
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("Context: {context}\n\nQuestion: {question}")
    ]

    prompt = ChatPromptTemplate.from_messages(messages)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return conversation_chain

# ==============================
# Display Chat
# ==============================

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Replace [ ... ] with $$...$$ explicitly
    def convert_brackets_to_mathjax(text):
        return re.sub(r'\[(.+?)\]', r'$$\1$$', text, flags=re.DOTALL)

    for i, message in enumerate(st.session_state.chat_history):
        content = convert_brackets_to_mathjax(message.content)
        
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", content), unsafe_allow_html=True)

# def handle_userinput(user_question):
#     start_time = time.time()  # <-- Start timing here

#     response = st.session_state.conversation({'question': user_question})

#     end_time = time.time()  # <-- End timing here
#     elapsed_time = end_time - start_time

#     # Save timing data
#     if 'query_times' not in st.session_state:
#         st.session_state.query_times = []
#     st.session_state.query_times.append(elapsed_time)

#     st.session_state.chat_history = response['chat_history']

#     # Replace [ ... ] with $$...$$ explicitly
#     def convert_brackets_to_mathjax(text):
#         return re.sub(r'\[(.+?)\]', r'$$\1$$', text, flags=re.DOTALL)

#     for i, message in enumerate(st.session_state.chat_history):
#         content = convert_brackets_to_mathjax(message.content)

#         if i % 2 == 0:
#             st.write(user_template.replace("{{MSG}}", content), unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace("{{MSG}}", content), unsafe_allow_html=True)

#     # Display timing
#     avg_time = sum(st.session_state.query_times) / len(st.session_state.query_times)
#     st.write(f"⏱️ Query Time: {elapsed_time:.2f}s | Average Query Time: {avg_time:.2f}s")

# ==============================
# Main App Logic
# ==============================
def main():
    load_dotenv()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('Chat with multiple PDFs :books:')
    user_question = st.text_input("Ask questions about the documents you provided")
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
                        st.error("No text found in uploaded PDFs. Make sure they are not scanned images or use OCR.")
                        return

                    text_chunks = get_chunks(raw_text)
                    if not text_chunks:
                        st.error("No text chunks generated. Check your PDFs.")
                        return

                    vector_store = get_vectorstore(text_chunks)
                    if vector_store is None:
                        st.error("Failed to create vector store.")
                        return

                    st.session_state.conversation = get_conversation_chain(vector_store)
                    st.success("PDFs processed successfully!")

if __name__ == '__main__':
    main()
