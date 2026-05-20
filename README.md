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

## Method Explanations

### `words_to_number(text)`

#### Purpose

Converts written number words into integers.

#### What It Does

The function:
- normalizes ordinal words,
- splits text into tokens,
- and computes the numeric value.

#### Examples

- `"three"` → `3`
- `"twenty one"` → `21`
- `"second"` → `2`

#### Why It Is Used

Used for interpreting user queries such as:
- `"solve the third problem"`
- `"first five questions"`

### `normalize_problem_reference(text)`

#### Purpose

Detects which problems or questions the user is referring to.

#### What It Does

The function supports:
- single problem references
- multiple problem references
- ranges
- ordinal references
- relative references such as `"last problem"`

#### Examples

- `"solve problem 3"`
- `"solve problems 2 to 5"`
- `"solve the second to last question"`

#### Output

Returns:
- reference type
- requested problem indices

### `extract_text_with_ocr(pdf_files)`

#### Purpose

Extracts text from scanned or image-based PDFs using OCR.

#### What It Does

The function:
1. converts PDF pages into images,
2. applies Tesseract OCR,
3. and combines extracted text.

#### Why It Is Used

Used when standard PDF text extraction fails.

### `get_text(pdf_docs)`

#### Purpose

Extracts text from uploaded PDF documents.

#### What It Does

The function:
- first attempts normal PDF text extraction using `PyPDF2`,
- and automatically switches to OCR when necessary.

#### OCR Fallback Conditions

OCR is used when:
- extracted text is too short,
- or scanned-document patterns are detected.

### `extract_all_problems(all_text)`

#### Purpose

Separates individual math problems from extracted document text.

#### What It Does

The function:
- normalizes whitespace,
- detects problem patterns using regex,
- and splits the document into individual problems.

#### Supported Formats

- `Problem 1`
- `Question 5`
- `1.)`

### `extract_keywords(text)`

#### Purpose

Extracts math-related keywords from user questions.

#### What It Does

The function searches for predefined math terms such as:
- derivative
- integral
- probability
- matrix
- vector
- logarithm

#### Why It Is Used

Used to improve semantic retrieval accuracy.

### `semantic_search_fallback(query, problems)`

#### Purpose

Finds relevant problems using semantic similarity search.

#### What It Does

The function:
1. generates embeddings using OpenAI embeddings,
2. stores vectors in FAISS,
3. performs similarity search,
4. and retrieves the most relevant problem.

#### Additional Feature

Math keywords are appended to improve retrieval quality.

### `find_exact_problem(idx, problems, raw_text=None)`

#### Purpose

Finds a problem using its explicit number.

#### What It Does

The function searches extracted problems using regex patterns such as:
- `Problem 3`
- `Question 3`
- `3.)`

#### Output

Returns the matched problem text if found.

### `clean_math_output(text)`

#### Purpose

Cleans generated math responses for plain-text display.

#### What It Does

The function:
- removes LaTeX syntax,
- converts fractions,
- converts square roots,
- and simplifies formatting.

#### Examples

- `\frac{1}{2}` → `(1/2)`
- `\sqrt{x}` → `√(x)`

### `handle_userinput(user_question)`

#### Purpose

Processes user questions and generates responses.

#### What It Does

The function:
1. parses the user query,
2. detects referenced problems,
3. retrieves matching context,
4. queries GPT-4o,
5. and formats the final response.

#### Supported Features

- exact problem retrieval
- semantic search
- multi-problem solving
- conversational retrieval

### `main()`

#### Purpose

Main entry point of the Streamlit application.

#### What It Does

The function:
- initializes session state,
- creates the Streamlit interface,
- handles PDF uploads,
- extracts document text,
- builds vector embeddings,
- initializes conversational retrieval,
- and manages user interaction.

#### User Interface Features

- PDF upload sidebar
- math question input
- conversational response display

## Overall System Workflow

```text
Upload PDFs
      ↓
Extract Text / OCR
      ↓
Detect Individual Problems
      ↓
Generate Embeddings
      ↓
Store in FAISS Vector Database
      ↓
User Question Input
      ↓
Problem Reference Detection
      ↓
Semantic Retrieval or Exact Matching
      ↓
GPT-4o Solution Generation
      ↓
Clean Plain-Text Math Output
      ↓
Display Final Answer
```
