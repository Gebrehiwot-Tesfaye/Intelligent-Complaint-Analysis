"""
app.py

Task 4: Interactive Chat Interface for RAG System using Gradio
- Allows users to ask questions about customer complaints
- Displays AI-generated answer
- Includes a clear button and supports chatbot-style conversation
"""
import sys
import os
import gradio as gr

# Ensure local src/ is in the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from rag_pipeline import rag_answer

def chat_fn(message, history):
    """
    Given a user message and chat history, runs the RAG pipeline to get an answer.
    Returns only the answer for display in the chatbot.
    """
    result = rag_answer(message)
    answer = result["answer"].strip()
    return answer, None

def respond(message, chat_history):
    """
    Handles a new user message: gets the RAG answer, appends to chat history, and returns updated state.
    Returns chat history in Gradio 'messages' format, with only the answer.
    """
    answer, _ = chat_fn(message, chat_history)
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer}
    ]
    return "", chat_history

def clear_chat():
    """
    Clears the chat input and history.
    """
    return "", []

with gr.Blocks() as demo:
    gr.Markdown("""
    # CrediTrust Complaint Insights Chatbot
    Ask any question about customer complaints. The AI will answer your question.
    """)
    chatbot = gr.Chatbot(type='messages')
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Type your question and press Enter")
        submit_btn = gr.Button("Submit")
        clear_btn = gr.Button("Clear")

    txt.submit(respond, [txt, chatbot], [txt, chatbot])
    submit_btn.click(respond, [txt, chatbot], [txt, chatbot])
    clear_btn.click(clear_chat, None, [txt, chatbot], queue=False)

if __name__ == "__main__":
    demo.launch(share=True)
