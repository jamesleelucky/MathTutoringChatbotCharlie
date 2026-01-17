import streamlit as st
import re
import os
import tempfile
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import nltk
from nltk.tokenize import sent_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# ==============================
# Load environment variables
# ==============================
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.error("Missing API key. Please set OPENAI_API_KEY in your .env file.")
    st.stop()

st.set_page_config(page_title='Chat with Multiple PDFs', page_icon=':books:')

# ==============================
# Download NLTK resources
# ==============================
nltk.download('punkt')

# ==============================
# Math Keywords for NER-like extraction
# ==============================
MATH_KEYWORDS = ["limit", "derivative", "integral", "sequence", "series", "probability",
                 "function", "logarithm", "exponential", "matrix", "vector", "sum"]

# ==============================
# Base number mappings
# ==============================
BASE_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeenth": 17, "eighteen": 18, "nineteen": 19
}
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
}
ORDINAL_TO_CARDINAL = {
    "first": "one", "second": "two", "third": "three", "fourth": "four",
    "fifth": "five", "sixth": "six", "seventh": "seven", "eighth": "eight",
    "ninth": "nine"
}

def words_to_number(text):
    text = text.replace("-", " ")
    parts = text.split()
    parts = [ORDINAL_TO_CARDINAL.get(word, word) for word in parts]
    num = 0
    for word in parts:
        if word in BASE_NUMBERS:
            num += BASE_NUMBERS[word]
        elif word in TENS:
            num += TENS[word]
    return num if num > 0 else None

# ==============================
# Detect problem/question reference
# ==============================
def normalize_problem_reference(text):
    text = text.lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\s,]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Handle 'last'
    if "last" in text:
        words = text.split()
        if words.count("last") == 1 and len(words) <= 4:
            return {"type": "relative", "values": ["last"]}
        if "last" in words:
            idx = words.index("last")
            if idx > 0:
                prev_word = words[idx - 1]
                if prev_word in ["second", "third", "fourth", "fifth"]:
                    offset = {"second": 1, "third": 2, "fourth": 3, "fifth": 4}.get(prev_word)
                    return {"type": "relative", "values": [offset]}
        match_last = re.search(r"(second|third|fourth|fifth)\s+to\s+last", text)
        if match_last:
            ordinal_word = match_last.group(1)
            offset = {"second": 1, "third": 2, "fourth": 3, "fifth": 4}.get(ordinal_word)
            return {"type": "relative", "values": [offset]}

    # Word-based ranges
    match_word_range = re.search(r"first\s+(\w+)\s+(problems?|questions?)", text)
    if match_word_range:
        word_num = match_word_range.group(1)
        end_num = BASE_NUMBERS.get(word_num, None)
        if end_num:
            return {"type": "range", "values": list(range(1, end_num + 1))}

    # Range detection
    match_range = re.search(r"(problem|question)s?\s*(\d+)\s*(to|-)\s*(\d+)", text)
    if match_range:
        start, end = int(match_range.group(2)), int(match_range.group(4))
        return {"type": "range", "values": list(range(start, end + 1))}

    # Multiple numbers
    normalized_text = text.replace(",", " ")
    match_numbers = re.findall(r"\d+", normalized_text)
    if len(match_numbers) > 1:
        return {"type": "multiple", "values": [int(x) for x in match_numbers]}

    # Explicit number
    match_number = re.search(r"(problem|question)?\s*#?\s*(\d+)", text)
    if match_number:
        return {"type": "number", "values": [int(match_number.group(2))]}

    # Ordinal suffix
    match_suffix = re.search(r"(\d+)(st|nd|rd|th)", text)
    if match_suffix:
        return {"type": "ordinal", "values": [int(match_suffix.group(1))]}

    # Word-based ordinals
    match_words = re.search(r"(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
                        r"eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|"
                        r"seventeenth|eighteenth|nineteenth|twentieth|thirtieth|fortieth|"
                        r"fiftieth|sixtieth|seventieth|eightieth|ninetieth|"
                        r"(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(\s+(one|two|three|four|five|six|seven|eight|nine|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth))?)", text)

    if match_words:
        num = words_to_number(match_words.group(0))
        if num:
            return {"type": "ordinal", "values": [num]}

    return None

# ==============================
# PDF text extraction
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
# Problem extraction with NLP
# ==============================
def extract_all_problems(all_text):
    all_text = re.sub(r"\s+", " ", all_text)  # Normalize spaces
    # Split using regex + tokenize
    regex_chunks = re.split(r"(?=(?:Problem\s*\d+|Question\s*\d+|\b\d+\s*[.)]))", all_text, flags=re.IGNORECASE)
    problems = [m.strip() for m in regex_chunks if re.match(r"^(Problem\s*\d+|Question\s*\d+|\d+\s*[.)])", m, flags=re.IGNORECASE)]
    # Sentence-level cleanup (optional)
    return problems

# ==============================
# Extract keywords from user query
# ==============================
def extract_keywords(text):
    keywords = []
    for kw in MATH_KEYWORDS:
        if kw in text.lower():
            keywords.append(kw)
    return keywords

# ==============================
# Semantic Fallback with keyword boost
# ==============================
def semantic_search_fallback(query, problems):
    if not problems:
        return None
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(problems, embedding=embeddings)
    # Boost score for keyword matches
    keywords = extract_keywords(query)
    search_query = query + " " + " ".join(keywords)
    result = vectorstore.similarity_search(search_query, k=1)
    return result[0].page_content if result else None

# ==============================
# Find problem by index or semantic similarity
# ==============================
def find_exact_problem(idx, problems, raw_text=None):
    pattern = rf"^(Problem\s*{idx}|Question\s*{idx}|{idx}\s*[.)])"
    for p in problems:
        if re.match(pattern, p, flags=re.IGNORECASE):
            return p
    return None

# ==============================
# Clean math output
# ==============================
def clean_math_output(text):
    text = re.sub(r'\$+', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'\\frac\{(.+?)\}\{(.+?)\}', r'(\1/\2)', text)
    text = re.sub(r'\\sqrt\{(.+?)\}', r'√(\1)', text)
    text = re.sub(r'([a-zA-Z0-9])\^(\d+)', r'\1^\2', text)
    text = re.sub(r'\{|\}', '', text)
    text = text.replace("\\", "")
    return text.strip()

# ==============================
# Handle user input
# ==============================
def handle_userinput(user_question):
    ref = normalize_problem_reference(user_question)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    if ref:
        values = ref["values"]
        total = len(st.session_state.get("problems_list", []))
        if ref["type"] == "relative" and "last" in values:
            values = [total]

        contexts = []
        for idx in values:
            if 0 < idx <= total:
                context = find_exact_problem(idx, st.session_state["problems_list"])
                if context:
                    contexts.append(f"Problem {idx}:\n{context}")
            else:
                # If index is invalid, ignore or note it
                contexts.append(f"Problem {idx}: Not found.")

        if contexts:
            combined_context = "\n\n".join(contexts)
            response = llm.predict(
                f"Context:\n{combined_context}\n\n"
                f"Solve each problem separately. For each problem:\n"
                f"- Show steps clearly\n"
                f"- Use plain text math (√ for roots, ^ for powers, / for fractions)\n"
                f"- End each with: Conclusion: ...\n"
                f"Do NOT use LaTeX."
            )
            clean_response = clean_math_output(response)
            st.markdown(f"### Solutions\n```text\n{clean_response}\n```")
        else:
            st.markdown("⚠ Could not find any of the requested problems.")
    else:
        # If no index, do semantic fallback
        context = semantic_search_fallback(user_question, st.session_state["problems_list"])
        if context:
            response = llm.predict(
                f"Context:\n{context}\n\nAnswer in plain text, show steps, end with Conclusion."
            )
            clean_response = clean_math_output(response)
            st.markdown(f"### Answer\n```text\n{clean_response}\n```")
        else:
            response = st.session_state.conversation({'question': user_question})
            clean_response = clean_math_output(response['chat_history'][-1].content)
            st.markdown(f"### Answer\n```text\n{clean_response}\n```")

# ==============================
# Main app
# ==============================
def main():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header('Chat with Multiple PDFs :books:')
    user_question = st.text_input("Ask a math question (e.g., 'solve problems 2,3 and 5').")

    if user_question:
        if st.session_state.conversation is None:
            st.warning("Please upload and process your PDFs first!")
        else:
            handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Upload Documents")
        pdf_docs = st.file_uploader("Upload text-based PDFs", accept_multiple_files=True)
        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_text(pdf_docs)
                    st.session_state["raw_text"] = raw_text
                    st.session_state["problems_list"] = extract_all_problems(raw_text)
                    # st.write("DEBUG: Extracted problems:", st.session_state["problems_list"])
                    chunks = RecursiveCharacterTextSplitter(
                        separators=["\nProblem", "\nQuestion", "\n\n", "\n", ".", " ", ""],
                        chunk_size=1200,
                        chunk_overlap=150
                    ).split_text(raw_text)
                    vectorstore = FAISS.from_texts(chunks, embedding=OpenAIEmbeddings())
                    st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                        llm=ChatOpenAI(model_name="gpt-4o", temperature=0),
                        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)
                    )
                    st.success("PDFs processed successfully!")

if __name__ == '__main__':
    main()
