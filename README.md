# CrediTrust Intelligent Complaint Analysis (RAG System)

A professional, end-to-end Retrieval-Augmented Generation (RAG) system for CrediTrust Financial to analyze customer complaints. This project enables internal teams to extract actionable insights from unstructured complaint narratives using state-of-the-art NLP and LLMs, with a public Gradio chat interface.

---

## Features

- **Data Preprocessing & EDA:** Cleans and prepares complaint data for semantic search.
- **Chunking & Embedding:** Splits narratives, generates embeddings, and indexes with ChromaDB for fast retrieval.
- **RAG Pipeline:** Combines robust retrieval, prompt engineering, and LLM-based answer generation (Flan-T5).
- **Interactive Chatbot:** Gradio-based UI for public/internal use, supporting real-time Q&A.
- **Documentation:** Comprehensive interim and final reports, with screenshots and analysis.

---

## Project Structure

```
Intelligent-Complaint-Analysis/
├── app.py                  # Gradio chat interface
├── data/                   # Raw and processed data
├── notebooks/              # EDA, preprocessing, and demo notebooks
├── reports/                # Interim and final reports, screenshots
├── src/                    # Core scripts: chunking, embedding, RAG pipeline
├── tests/                  # (Optional) Unit tests
├── vector_store/           # Persisted ChromaDB vector store
├── requirements.txt        # Python dependencies
└── README.md               # Project overview and instructions
```

---

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Intelligent-Complaint-Analysis
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare data:**
   - Place raw complaint data in `data/`.
   - Run EDA/preprocessing notebook: `notebooks/eda_preprocessing.ipynb`.
   - Run chunking/embedding script: `src/chunk_and_embed.py` (creates/updates `vector_store/`).
4. **Run the Gradio app:**
   ```bash
   python app.py
   ```
   - The app will launch in your browser. Use `share=True` for public links.

---

## Usage

- Ask any question about customer complaints in the chat interface.
- The AI will retrieve relevant complaint excerpts and generate a grounded answer.
- (Optional) Toggle sources for transparency.
- For demonstration, see screenshots/GIFs in `reports/`.

---

## Reports & Documentation

- **Interim Report:** `reports/interim_report.md`
- **Final Report:** `reports/final_report.md`
- **Notebooks:** EDA, preprocessing, and demo in `notebooks/`

---

## Recommendations & Next Steps

- Integrate more advanced LLMs (e.g., OpenAI GPT-4).
- Add streaming responses and user feedback loop.
- Quantitative evaluation and continuous improvement.

---

## License

MIT License. See `LICENSE` for details.

---

## Acknowledgements

- Hugging Face, LangChain, ChromaDB, Gradio, and the open-source community.
- CrediTrust Financial for project support and feedback.
