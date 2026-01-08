"""
NexusAI - RAG Chain Module (Fixed for LangChain 0.1.20)
Advanced Retrieval-Augmented Generation System
Uses OpenAI LLM and OpenAI Embeddings for compatibility
Developed by: Muneeb
Supervised by: Hamza Arif
"""

import os
import faiss
from dotenv import load_dotenv

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.chat_message_histories import ChatMessageHistory

# Use OpenAI LLM and embeddings compatible with LangChain 0.1.20
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# =======================
# LOAD ENV VARIABLES
# =======================

load_dotenv()

# =======================
# SESSION STORE
# =======================

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Create or return chat history for a session"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# =======================
# PDF LOADING
# =======================

def load_pdf_from_user(pdf_path: str):
    print(f"[INFO] Loading PDF: {pdf_path}")
    loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()
    print(f"[INFO] Loaded {len(docs)} pages")
    return docs

# =======================
# DOCUMENT SPLITTING
# =======================

def split_documents(docs):
    print("[INFO] Splitting documents...")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="gpt2",
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = splitter.split_documents(docs)
    print(f"[INFO] Created {len(splits)} chunks")
    return splits

# =======================
# VECTOR STORE
# =======================

def create_vector_store(splits):
    print("[INFO] Creating vector store...")

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Test embedding dimension
    test_embedding = embeddings.embed_query("dimension check")
    embedding_dimension = len(test_embedding)

    index = faiss.IndexFlatL2(embedding_dimension)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vector_store.add_documents(splits)
    print(f"[INFO] Stored {len(splits)} embeddings")

    return vector_store

# =======================
# BUILD RAG CHAIN
# =======================

def build_conversational_rag_chain(retriever):
    print("[INFO] Building Conversational RAG Chain...")

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3,
        max_tokens=300
    )

    # In LangChain 0.1.20, trim_messages is not available
    base_chain = llm

    # -------- STEP 1: Contextualize Question --------
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, "
         "rewrite the question as a standalone query without answering it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        base_chain,
        retriever,
        contextualize_prompt
    )

    # -------- STEP 2: Answer Question --------
    qa_system_prompt = """
You are NexusAI, an advanced AI assistant specialized in document intelligence.

Rules:
- Answer ONLY using the provided context
- If the answer is not in the document, say so clearly
- Keep answers concise (3–7 sentences)
- Maintain conversational continuity

Document Context:
{context}
"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    document_chain = create_stuff_documents_chain(
        base_chain,
        qa_prompt
    )

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        document_chain
    )

    # -------- STEP 3: Attach Memory --------
    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    print("[INFO] RAG Chain READY")
    return conversational_chain

# =======================
# LOCAL TESTING
# =======================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rag_chain.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    docs = load_pdf_from_user(pdf_path)
    splits = split_documents(docs)
    vector_store = create_vector_store(splits)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    rag_chain = build_conversational_rag_chain(retriever)

    session_id = "test_session"

    print("\nNexusAI Ready. Ask questions (type 'quit' to exit)\n")

    while True:
        q = input("Q: ")
        if q.lower() in ["quit", "exit"]:
            break

        res = rag_chain.invoke(
            {"input": q},
            config={"configurable": {"session_id": session_id}}
        )

        print(f"A: {res['answer']}\n")
