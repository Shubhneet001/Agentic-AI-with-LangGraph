from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import os
import requests
import aiosqlite
import httpx
import sqlite3
import asyncio
import threading


load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "My Chatbot"
groq_api_key = os.getenv("GROQ_API_KEY")
alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=groq_api_key
)

# Tools
search_tool = TavilySearch(max_results=5, topic="general")

@tool
async def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock quote from Alpha Vantage."""

    if not alpha_vantage_key:
        return {"error": "ALPHA_VANTAGE_API_KEY is not configured"}

    url = "https://www.alphavantage.co/query"

    # Async HTTP request: doesn't block the event loop while waiting.
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            url,
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": alpha_vantage_key
            }
        )
        response.raise_for_status()
        return response.json()

client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": "python3",
            "args": ["C:/Users/p/Documents/Coding/Agentic-AI-with-LangGraph/chatbot/chatbot-mcp/cal_server.py"]   # local server
        },
        "expense": {
            "transport": "streamable_http",  # if this fails, try "sse"
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"    # remote server
        }
    }
)

def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []

mcp_tools = load_mcp_tools()
tools = [search_tool, get_stock_price, *mcp_tools]
llm = llm.bind_tools(tools)

# Thread metadata database (kept synchronous)
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS threads(
    thread_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# Chatbot graph
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call"""
    messages = state["messages"]
    response = await llm.ainvoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools) if tools else None

async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)

checkpointer = run_async(_init_checkpointer())

# Graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# Thread metadata functions
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

    return [
        {
            "id": row[0],
            "title": row[1],
            "created_at": row[2]
        }
        for row in cursor.fetchall()
    ]

    
