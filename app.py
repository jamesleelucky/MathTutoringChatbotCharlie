import streamlit as st
import re
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

st.set_page_config(page_title='Chat with Multiple PDFs', page_icon=':books:')

# ==============================
# Ordinal Mapping for words
# ==============================
ORDINAL_MAP = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10
}

def normalize_problem_reference(text):
    text = text.lower()

    # Check for ordinal words like first, second
    for word, num in ORDINAL_MAP.items():
        if word in text:
            return {"type": "ordinal", "value": num}

    # Check for numeric ordinals: 1st, 2nd, 3rd, 23rd, etc.
    match_suffix = re.search(r"(\d+)(st|nd|rd|th)", text)
    if match_suffix:
        return {"type": "ordinal", "value": int(match_suffix.group(1))}

    # Check for explicit "problem/question X" or "#X"
    match_number = re.search(r"(problem|question)?\s*#?\s*(\d+)", text)
    if match_number:
        return {"type": "number", "value": match_number.group(2)}

    return None

# ==============================
# OCR Extraction
# ==============================
def extract_text_with_ocr(pdf_files):
    text = ""
    for uploaded_file in pdf_files:
        uploaded_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        images = convert_from_path(tmp_path, dpi=300)
        for img in images:
            text += pytesseract.image_to_string(img, lang="eng", config="--oem 1 --psm 6") + "\n"

        os.remove(tmp_path)
    return text

# ==============================
# Extract text from PDFs
# ==============================
def get_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

    if len(text.strip()) < 100 or "____" in text:
        text = extract_text_with_ocr(pdf_docs)
    return text

# ==============================
# Extract all problems from text
# ==============================
def extract_all_problems(all_text):
    matches = re.split(r"(?=Problem\s*\d+)", all_text)
    problems = [m.strip() for m in matches if m.strip().startswith("Problem")]
    return problems

def find_exact_problem(ref, problems):
    if not problems:
        return None

    if ref["type"] == "number":
        # Exact problem number match
        for p in problems:
            if re.search(rf"Problem\s*{ref['value']}\b", p):
                return p
    else:
        # Ordinal match → nth problem in order
        index = ref["value"] - 1
        if 0 <= index < len(problems):
            return problems[index]
    return None

# ==============================
# Chunking & Filtering
# ==============================
def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\nProblem", "\nQuestion", "\n\n", "\n", ".", " ", ""],
        chunk_size=1200,
        chunk_overlap=150
    )
    return splitter.split_text(text)

def filter_chunks(chunks):
    filtered = [ch for ch in chunks if ch.count("Problem") <= 3 and ch.count("Question") <= 1]
    return filtered if filtered else chunks

# ==============================
# Vector Store
# ==============================
def get_vectorstore(text_chunks):
    if not text_chunks:
        st.error("No valid text chunks found. Check your PDF content.")
        return None
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# ==============================
# Translate to English
# ==============================
def translate_to_english(text):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    return llm.predict(f"Translate to English, keep math symbols unchanged:\n\n{text}")

# ==============================
# Clean LaTeX Output
# ==============================
def clean_math_output(text):
    text = re.sub(r'\$+', '', text)
    text = re.sub(r'\\frac\{(.+?)\}\{(.+?)\}', r'(\1/\2)', text)
    text = re.sub(r'\\sqrt\{(.+?)\}', r'√(\1)', text)
    text = re.sub(r'([a-zA-Z0-9])\^(\d+)', r'\1^\2', text)
    text = text.replace("\\", "")
    return text

# ==============================
# Conversation Chain
# ==============================
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    system_prompt = """
    You are an expert math tutor. Always respond in ENGLISH.
    Answer ONLY the user's question. Ignore other questions in the context.
    Do NOT use LaTeX, MathJax, or $ signs. Do NOT use \commands.
    Use plain text math symbols: √, ∫, Σ, →, x² or x^2.
    Show all steps clearly, one per line.
    Begin with the original question, then the solution.
    End with "Conclusion: ..." and the final answer.
    """

    messages = [
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("""
        Context (for reference only, ignore other questions):
        {context}

        User Question:
        {question}
        """)
    ]

    prompt = ChatPromptTemplate.from_messages(messages)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# ==============================
# Handle User Input
# ==============================
def handle_userinput(user_question):
    translated_question = translate_to_english(user_question)
    ref = normalize_problem_reference(translated_question)
    response_text = ""

    if ref and "problems_list" in st.session_state:
        exact_context = find_exact_problem(ref, st.session_state["problems_list"])
        if exact_context:
            llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
            response_text = llm.predict(
                f"Context (problem to solve):\n{exact_context}\n\n"
                f"Solve ONLY the problem in the context above. Ignore the wording of the user question. "
                f"Provide a detailed, step-by-step solution in plain text math (no LaTeX, no $ signs). "
                f"Use √, Σ, ∫, and x^2 for powers. End with 'Conclusion: ...'."
            )
        else:
            response_text = "Could not find the problem in the document."
    else:
        # For general conceptual questions → fallback to FAISS
        response = st.session_state.conversation({'question': translated_question})
        response_text = response['chat_history'][-1].content

    if not re.search(r'[a-zA-Z]', response_text):
        response_text = translate_to_english(response_text)

    response_text = clean_math_output(response_text)
    st.write(bot_template.replace("{{MSG}}", response_text), unsafe_allow_html=True)

# ==============================
# Main App
# ==============================
def main():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header('Chat with Multiple PDFs :books:')

    user_question = st.text_input("Ask a complete text-based math question. ")
    if user_question:
        if st.session_state.conversation is None:
            st.warning("Please upload and process your PDFs first!")
        else:
            handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Upload Documents")
        pdf_docs = st.file_uploader("Upload ONLY text-based PDFs", accept_multiple_files=True)
        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No text found. Make sure the PDFs have complete text or use OCR.")
                        return

                    # Extract problems list for ordinal/numeric reference
                    problems_list = extract_all_problems(raw_text)
                    st.session_state["problems_list"] = problems_list

                    # Prepare vectorstore for retrieval
                    text_chunks = get_chunks(raw_text)
                    filtered_chunks = filter_chunks(text_chunks)
                    vector_store = get_vectorstore(filtered_chunks)
                    if vector_store:
                        st.session_state.conversation = get_conversation_chain(vector_store)
                        st.success("PDFs processed successfully!")

if __name__ == '__main__':
    main()
