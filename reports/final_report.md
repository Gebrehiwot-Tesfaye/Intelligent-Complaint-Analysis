# Final Report: Intelligent Complaint Analysis for CrediTrust Financial

**Prepared by: Gebrehiwot Tesfaye**

## Project Overview

CrediTrust Financial receives thousands of customer complaints monthly across five major product categories. To empower internal teams with actionable insights, we developed a professional, end-to-end Retrieval-Augmented Generation (RAG) system. This system enables natural language querying of unstructured complaint narratives, surfacing relevant information and generating concise, context-grounded answers.

## Task 1: Exploratory Data Analysis (EDA) and Preprocessing

### Objective

- Understand the structure, content, and quality of the complaint data.
- Prepare the data for downstream RAG processing.

### Steps & Findings

- **Data Loading & Overview:** Loaded the CFPB complaints dataset and examined its structure, columns, and sample records. Found a wide range of narrative lengths and product categories.
- **Distribution Analysis:** Visualized complaint counts by product and narrative length. Identified that some products are more represented and that narratives vary from very short to very long.
- **Missing Data Handling:** Removed records with missing or empty narratives, ensuring only meaningful complaints are retained.
- **Filtering:** Focused on five target products: Credit card, Personal loan, Buy Now Pay Later, Savings account, Money transfer. This aligns the dataset with business priorities.
- **Text Cleaning:** Lowercased text, removed special characters, boilerplate phrases, and extra spaces. This improved consistency and embedding quality.
- **Output:** Cleaned and filtered data saved as `data/filtered_complaints.csv`. All EDA and preprocessing steps are documented in `notebooks/eda_preprocessing.ipynb`.

## Task 2: Text Chunking, Embedding, and Vector Store Indexing

### Objective

- Convert complaint narratives into a format suitable for efficient semantic search and retrieval.

### Steps & Findings

- **Chunking:** Used LangChain's `RecursiveCharacterTextSplitter` with `chunk_size=300` and `chunk_overlap=50` to split long narratives into overlapping chunks, preserving context across boundaries.
- **Embedding:** Used `sentence-transformers/paraphrase-MiniLM-L3-v2` for fast, high-quality semantic embeddings. The model is lightweight and CPU-friendly, ideal for local deployment.
- **Indexing:** Embedded chunks and metadata (complaint ID, product) were stored in a ChromaDB vector store (`vector_store/`). This enables fast similarity search and traceability to original complaints.
- **Batch Processing:** Optimized for local CPU by processing in batches, improving speed and scalability.
- **Output:** Script: `src/chunk_and_embed.py`. Vector store: `vector_store/`.

## Task 3: RAG Core Logic and Evaluation

### Objective

- Build a robust RAG pipeline for question answering over indexed complaints, with prompt engineering and LLM-based answer generation.

### Steps & Findings

- **Retriever:** Embeds user question and retrieves top-k relevant complaint chunks from the vector store using semantic similarity.
- **Prompt Engineering:** Uses a robust template instructing the LLM to act as a financial analyst assistant, answer only from provided context, and state if information is insufficient.
- **Generator:** Combines the prompt, user question, and retrieved context, then sends to an instruction-tuned LLM (`google/flan-t5-base` or `flan-t5-large`) for answer generation.
- **Evaluation:** Performed qualitative evaluation using representative questions. See table below for details.

#### Prompt Template Example

```
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.
Context: {context}
Question: {question}
Answer:
```

### Qualitative Evaluation

| Question                                                | Generated Answer                                                                                                                                              | Retrieved Sources                                                                                                                                      | Quality Score (1-5) | Comments/Analysis                                                                      |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------- | -------------------------------------------------------------------------------------- |
| Why are people unhappy with BNPL?                       | Many customers are dissatisfied with Buy Now Pay Later (BNPL) due to unexpected fees, unclear repayment terms, and issues with refunds or payment processing. | [BNPL] "i was charged a late fee even though i paid on time."\n[BNPL] "the repayment schedule was not clearly explained and i incurred extra charges." | 5                   | Answers are specific, grounded in retrieved complaints, and highlight key pain points. |
| What are the most common complaints about credit cards? | The most common complaints about credit cards involve high interest rates, billing errors, and poor customer service.                                         | [Credit card] "my interest rate increased without notice."\n[Credit card] "customer service was unhelpful when disputing a charge."                    | 5                   | Captures main themes and uses direct complaint excerpts.                               |
| Are there issues with money transfers?                  | Yes, customers report delays, lost funds, and high transfer fees as major issues with money transfers.                                                        | [Money transfer] "my transfer took over a week to process."\n[Money transfer] "i was charged more than the advertised fee."                            | 4                   | Good coverage of issues, but some details may be missing if context is limited.        |
| How do customers feel about personal loans?             | Customers often express frustration with personal loans due to hidden fees, confusing terms, and slow approval processes.                                     | [Personal loan] "the loan terms were not clearly explained."\n[Personal loan] "approval took much longer than promised."                               | 4                   | Answers are relevant and context-based, but could be improved with more examples.      |
| What problems do users report with savings accounts?    | Users commonly report issues with account access, unexpected fees, and slow customer support for savings accounts.                                            | [Savings account] "i was locked out of my account without explanation."\n[Savings account] "unexpected maintenance fees were charged."                 | 5                   | Directly addresses user concerns with clear, retrieved examples.                       |

#### Analysis

- The retriever surfaces relevant complaint excerpts for most questions.
- Prompt template keeps the LLM grounded in context.
- Some answers may be generic if context is insufficient; further prompt tuning or a more advanced LLM may help.
- Future improvements: try more powerful LLMs, experiment with prompt styles, add quantitative evaluation.

---

## Task 4: Interactive Chat Interface

### Objective

- Provide an intuitive, public-facing chat interface for internal teams to query the RAG system.

### Steps & Findings

- **Implementation:** Built with Gradio (`app.py`), featuring a chatbot UI with text input, submit/clear buttons, and answer display.
- **Integration:** Connected to the RAG pipeline for real-time, context-grounded answers.
- **Features:** Handles multi-turn conversation, chat history, robust error handling, and user feedback. Option to display retrieved sources (toggleable). Public link enabled for easy access and demonstration.
- **Output:** Gradio app: `app.py`.

## Project Structure

- `data/`: Raw and processed data.
- `src/`: All core scripts (chunking, embedding, RAG pipeline).
- `vector_store/`: Persisted ChromaDB vector database.
- `app.py`: Gradio chat interface.
- `notebooks/`: EDA and development notebooks.
- `reports/`: Documentation (interim and final reports).
- `tests/`: Unit and integration tests.

## Key Learnings & Recommendations

- RAG systems can unlock actionable insights from unstructured complaints, improving customer experience and operational efficiency.
- Careful preprocessing and chunking are critical for high-quality retrieval.
- Prompt engineering and LLM choice significantly impact answer quality.
- For production, consider:
  - More advanced LLMs (e.g., GPT-4, Claude).
  - Quantitative evaluation (e.g., answer accuracy, latency).
  - User feedback loop for continuous improvement.

## References

- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB](https://www.trychroma.com/)
- [Gradio](https://gradio.app/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
