import tempfile
from typing import TypedDict, Literal, Annotated, Any, Dict, Optional
import sqlite3
import requests
import random
import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool


os.environ['LANGCHAIN_PROJECT'] = 'My Chatbot'


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise RuntimeError(
        f"GROQ_API_KEY not found."
    )

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=groq_api_key
)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# RAG

# pdf retriever store (per thread)
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available"""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded pdf and store it in the thread.
    Returns a summary dict that can be surfaced in the UI
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass

# Tools
search_tool = TavilySearch(
    max_results=5,
    topic="general"
)

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """

    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using ALpha Vantage with API key in the URL
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


# bind tools with LLM
tools = [get_stock_price, search_tool, calculator, rag_tool]
llm = llm.bind_tools(tools)

# -------- Database Setup --------
conn = sqlite3.connect("chatbot.db", check_same_thread=False)      # false to use multiple threads in the same database
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS threads(
    thread_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()

# -------- Chatbot Graph --------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState, config=None):
    """LLM that may answer or request a tool call"""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )
    messages = [system_message, *state["messages"]]
    response = llm.invoke(messages, config=config)
    return {'messages': [response]}

tool_node = ToolNode(tools)


checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

# nodes
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

# edges
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

# -------- Thread Metadata --------

def create_thread(thread_id, title="New Chat"):
    cursor.execute(
        "INSERT OR IGNORE INTO threads(thread_id, title) VALUES(?, ?)",
        (str(thread_id), title)
    )
    conn.commit()

def update_thread_title(thread_id, title):
    cursor.execute(
        "UPDATE threads SET title=? WHERE thread_id=?",
        (title, str(thread_id))
    )
    conn.commit()

def retrieve_all_threads():
    cursor.execute("""
        SELECT thread_id, title, created_at
        FROM threads
        ORDER BY created_at DESC
""")

    rows = cursor.fetchall()

    return [
        {
            "id": row[0],
            "title": row[1],
            "created_at": row[2]
        }
        for row in rows
    ]

# def retrieve_all_threads():
#     all_threads = set()
#     for checkpoint in checkpointer.list(None):
#         all_threads.add(checkpoint.config['configurable']['thread_id'])

#     return list(all_threads)