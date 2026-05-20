# MathTutoringChatbotCharlie

An AI-powered PDF math assistant built with Streamlit, LangChain, OpenAI, and FAISS that extracts, retrieves, and solves math problems from uploaded PDF documents.

The application can:
- detect numbered math problems automatically
- solve referenced problems such as:
  - "solve problem 3"
  - "solve problems 2 to 5"
  - "solve the last question"
- retrieve semantically similar problems using embeddings
- extract text from scanned PDFs using OCR
- maintain conversational memory across questions

The system uses OpenAI embeddings, FAISS vector search, LangChain retrieval pipelines, and GPT-4o for contextual problem solving.

## How It Works
- User uploads PDFs
- Text is extracted from documents
- OCR is applied if necessary
- Text is split into chunks
- Embeddings are generated using OpenAI
- FAISS stores vector embeddings
- User questions are matched semantically
- GPT-4o generates contextual answers

## Future Improvements
- Support for image uploads
- Better math rendering
- Multi-language OCR
- Export chat history
