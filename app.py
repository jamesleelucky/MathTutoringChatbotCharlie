import streamlit as st
import sys
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
def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
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
        You are the best math tutor in the world. Solve ONLY the given question using the provided context.
        Do NOT invent problems that are not in the question.


        Format your response as:

        Solution:
        1. Explain what the equation is. State the original equation clearly in normal Math expressions, not LeTex commands.

        2. Show the steps to solve the problem. DO NOT use raw LaTeX like \equiv and \pmod in the explanation. Use plain English text with normal Math expressions instead. DO NOT use raw LaTeX like \equiv and \pmod in the explanation. Use plain English text with normal Math expressions instead.DO NOT use raw LaTeX like \equiv and \pmod in the explanation. Use plain English text with normal Math expressions instead.

        3. Conclusion:
            State final result or that no solution exists.

        Rules:
        - Use clear English with MathJax for math.
        - Do NOT approximate unless required.
        - No LaTeX code outside MathJax blocks.
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

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

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

# import streamlit as st
# import sys
# import time
# import csv
# import os
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from templates import css, bot_template, user_template

# st.set_page_config(page_title='Chat with multiple PDFs', page_icon=':books:')

# # ==============================
# # PDF Text Extraction
# # ==============================
# def get_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             extracted = page.extract_text()
#             if extracted:
#                 text += extracted
#     return text

# # ==============================
# # Text Chunking (optimized)
# # ==============================
# def get_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator='\n',
#         chunk_size=500,   # Reduced for speed
#         chunk_overlap=100,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# # ==============================
# # Vector Store (with caching)
# # ==============================
# def get_vectorstore(text_chunks):
#     embeddings = OpenAIEmbeddings()
#     index_path = "faiss_index"

#     if os.path.exists(index_path):
#         return FAISS.load_local(index_path, embeddings)
#     else:
#         vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#         vectorstore.save_local(index_path)
#         return vectorstore

# # ==============================
# # Improved Prompt with Examples
# # ==============================
# def get_system_prompt():
#     return """
#     You are the best math tutor in the world. Solve ONLY the given question using the provided context.
#     Do NOT invent problems that are not in the question.

#     Strictly follow this format:

#     Solution:
#     1. Restate the question in plain English.
#     2. Show each step in simple language and math notation (no raw LaTeX).
#     3. Conclusion: State the final answer clearly in one sentence.

#     Rules:
#     - Do NOT include unnecessary steps or unrelated math.
#     - Use clear, concise English suitable for high school level.
#     - No LaTeX commands like \\equiv or \\pmod.
#     - Use MathJax for inline math.

#     Example:
#     Question: Solve x + 2 = 5
#     Solution:
#     1. The equation is x + 2 = 5.
#     2. Subtract 2 from both sides: x = 3.
#     3. Conclusion: x = 3.
#     """

# # ==============================
# # Conversation Chain (with GPT-3.5-turbo)
# # ==============================
# def get_conversation_chain(vectorstore):
#     llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, streaming=True)

#     system_prompt = get_system_prompt()

#     messages = [
#         SystemMessagePromptTemplate.from_template(system_prompt),
#         HumanMessagePromptTemplate.from_template("Context: {context}\n\nQuestion: {question}")
#     ]

#     prompt = ChatPromptTemplate.from_messages(messages)
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory,
#         combine_docs_chain_kwargs={"prompt": prompt}
#     )
#     return conversation_chain

# # ==============================
# # Handle User Input with Metrics
# # ==============================
# def handle_userinput(user_question):
#     start_time = time.time()

#     response = st.session_state.conversation({'question': user_question})

#     end_time = time.time()
#     resolution_time = round(end_time - start_time, 2)

#     # Validate output format
#     answer = response['chat_history'][-1].content
#     if not all(section in answer for section in ["Solution:", "Conclusion:"]):
#         answer += "\n\nNote: The answer did not fully follow the required format."

#     # Display
#     st.session_state.chat_history = response['chat_history']
#     for i, message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

#     st.write(f"⏱ Query resolved in {resolution_time} seconds")

#     if "metrics" not in st.session_state:
#         st.session_state.metrics = []
#     st.session_state.metrics.append({"query": user_question, "time": resolution_time})

#     # Clarity feedback
#     clarity_score = st.slider("Rate clarity of the response (1=Poor, 5=Excellent)", 1, 5, 3)
#     if "clarity_ratings" not in st.session_state:
#         st.session_state.clarity_ratings = []
#     st.session_state.clarity_ratings.append(clarity_score)

#     st.write(f" Clarity score recorded: {clarity_score}/5")

# # ==============================
# # Export Metrics
# # ==============================
# def export_metrics():
#     if "metrics" in st.session_state and st.session_state.metrics:
#         with open("metrics.csv", "w", newline="") as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(["Query", "Resolution Time (s)"])
#             for item in st.session_state.metrics:
#                 writer.writerow([item["query"], item["time"]])

#         avg_time = sum(item["time"] for item in st.session_state.metrics) / len(st.session_state.metrics)
#         avg_clarity = sum(st.session_state.clarity_ratings) / len(st.session_state.clarity_ratings)
#         st.write(f" Average Query Resolution Time: {avg_time:.2f} seconds")
#         st.write(f" Average Clarity Score: {avg_clarity:.2f}/5")

#         st.download_button("Download Metrics CSV", open("metrics.csv", "rb"), "metrics.csv")

# # ==============================
# # Main App Logic
# # ==============================
# def main():
#     load_dotenv()

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None

#     st.header('Chat with multiple PDFs :books:')
#     user_question = st.text_input("Ask questions about the documents you provided")
#     if user_question:
#         if st.session_state.conversation is None:
#             st.warning("Please upload and process your PDFs first!")
#         else:
#             handle_userinput(user_question)

#     with st.sidebar:
#         st.subheader("Your Documents")
#         pdf_docs = st.file_uploader("Upload your PDFs here and click 'Process'", accept_multiple_files=True)
#         process_button = st.button("Process")
#         st.write("---")
#         if st.button("Export Metrics"):
#             export_metrics()

#         if process_button:
#             if not pdf_docs:
#                 st.error("Please upload at least one PDF.")
#             else:
#                 with st.spinner("Processing..."):
#                     raw_text = get_text(pdf_docs)
#                     if not raw_text.strip():
#                         st.error("No text found in uploaded PDFs. Make sure they are not scanned images or use OCR.")
#                         return

#                     text_chunks = get_chunks(raw_text)
#                     if not text_chunks:
#                         st.error("No text chunks generated. Check your PDFs.")
#                         return

#                     vector_store = get_vectorstore(text_chunks)
#                     if vector_store is None:
#                         st.error("Failed to create vector store.")
#                         return

#                     st.session_state.conversation = get_conversation_chain(vector_store)
#                     st.success("PDFs processed successfully!")

# if __name__ == '__main__':
#     main()
