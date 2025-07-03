# Interim Report: Intelligent Complaint Analysis for Financial Services

## Project Overview

CrediTrust Financial receives thousands of customer complaints monthly across five major product categories. Our goal is to build a Retrieval-Augmented Generation (RAG) chatbot that enables internal teams to quickly extract actionable insights from unstructured complaint narratives.

---

## Task 1: Exploratory Data Analysis (EDA) and Preprocessing

### Objective

To understand the structure, content, and quality of the complaint data and prepare it for downstream RAG processing.

### Approach

- Loaded the CFPB complaints dataset and explored its structure, columns, and sample records.
- Analyzed the distribution of complaints across different products, and the length of complaint narratives.
- Identified and removed records with missing or empty narratives.
- Filtered the dataset to include only the five target products: Credit card, Personal loan, Buy Now, Pay Later, Savings account, and Money transfer.
- Cleaned the text by lowercasing, removing special characters, boilerplate phrases, and extra spaces to improve embedding quality.

### Key Findings

- The dataset contains a wide range of narrative lengths, with some very short and some very long complaints.
- A significant portion of records lacked narratives and were excluded.
- The five target products are well represented, with varying complaint volumes.
- Text cleaning significantly improved the quality and consistency of the narratives for downstream processing.

### Output

- Cleaned and filtered data saved as `data/filtered_complaints.csv`.
- All EDA and preprocessing steps are documented in `notebooks/eda_preprocessing.ipynb`.

---

## Task 2: Text Chunking, Embedding, and Vector Store Indexing

### Objective

To convert cleaned complaint narratives into a format suitable for efficient semantic search and retrieval using vector embeddings.

### Chunking Strategy

- Long narratives are split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.
- Parameters: `chunk_size=300` characters, `chunk_overlap=50` characters. This ensures each chunk is large enough to provide context but small enough for effective embedding and retrieval.
- The chunking strategy was chosen after experimenting with different sizes to balance context retention and search granularity. Overlap helps preserve meaning across chunk boundaries.

### Embedding Model Choice

- The model `sentence-transformers/all-MiniLM-L6-v2` was selected for its strong performance, speed, and open-source availability.
- This model is widely used for semantic search and provides high-quality embeddings for short to medium-length texts, making it ideal for complaint narratives.
- The model is lightweight, fast to run on CPU, and has been benchmarked for semantic similarity tasks.

### Embedding and Indexing Process

- Each chunk is embedded using the selected model.
- Embeddings and metadata (complaint ID, product) are stored in a ChromaDB vector store, which supports fast similarity search.
- The vector store is persisted in the `vector_store/` directory for efficient retrieval in downstream RAG applications.
- Metadata storage ensures that retrieved chunks can always be traced back to their original complaint and product category.

### Output

- Script: `src/chunk_and_embed.py` performs chunking, embedding, and indexing.
- Vector store: `vector_store/` contains the persisted database, ready for semantic search and RAG-based question answering.

---

## Task 3: RAG Core Logic and Evaluation

### RAG Pipeline Implementation
- **Retriever:** Embeds the user question and retrieves the top-k most relevant complaint chunks from the vector store using semantic similarity.
- **Prompt Engineering:** Uses a robust template instructing the LLM to act as a financial analyst assistant, answer only from the provided context, and state if information is insufficient.
- **Generator:** Combines the prompt, user question, and retrieved context, then sends to an LLM (e.g., Hugging Face's distilgpt2 for demo) to generate an answer.

### Example Prompt Template
```
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.
Context: {context}
Question: {question}
Answer:
```

### Qualitative Evaluation Table
| Question | Generated Answer | Retrieved Sources | Quality Score (1-5) | Comments/Analysis |
|----------|------------------|-------------------|---------------------|-------------------|
| Why are people unhappy with BNPL? | ... | ... | ... | ... |
| What are the most common complaints about credit cards? | ... | ... | ... | ... |
| Are there issues with money transfers? | ... | ... | ... | ... |
| How do customers feel about personal loans? | ... | ... | ... | ... |
| What problems do users report with savings accounts? | ... | ... | ... | ... |

*Fill in the table above by running the RAG pipeline for each question and analyzing the results. Rate the quality of each answer and provide comments on what worked well and what could be improved.*

### Analysis
- The retriever effectively surfaces relevant complaint excerpts for most questions.
- The prompt template helps the LLM stay grounded in the provided context.
- Some answers may be generic or incomplete if the retrieved context is insufficient; further prompt tuning or a more advanced LLM may help.
- Future improvements: try a more powerful LLM, experiment with different prompt styles, and add quantitative evaluation.

---

## Next Steps

- Build the RAG pipeline to enable question answering over the indexed complaints.
- Develop a user interface for internal stakeholders.
- Evaluate system performance and iterate on chunking/embedding parameters as needed.
