from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

import sqlite3
import requests
import random
import os

os.environ['LANGCHAIN_PROJECT'] = 'My Chatbot'

load_dotenv()

# LLM Setup
llm = ChatGroq(model="openai/gpt-oss-20b")

# Tools
search_tool = DuckDuckGoSearchRun(region="us-en")

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

# bind tools with LLM
tools = [get_stock_price, search_tool, calculator]
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

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
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