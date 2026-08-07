from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

# LLM Setup
llm = ChatGroq(model="llama-3.1-8b-instant")

# -------- Database Setup --------
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

# -------- Chatbot Graph --------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

# conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)    # false to use multiple threads in the same database

checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

# nodes
graph.add_node('chat_node', chat_node)

# edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

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