import streamlit as st
from backend import (
    chatbot,
    create_thread,
    llm,
    retrieve_all_threads,
    update_thread_title
)
from langchain_core.messages import HumanMessage, AIMessage
import uuid    # for generating multiple random thread IDs
from datetime import datetime


# **************************** Utility Functions ****************************

# generates a new random thread ID
def generate_thread_id():
    thread_id = uuid.uuid4()
    return str(thread_id)

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    create_thread(thread_id)
    st.session_state["chat_threads"] = retrieve_all_threads()
    st.session_state['message_history'] = []

# def add_thread(thread_id):
#     exists = any(thread['id'] == thread_id for thread in st.session_state['chat_threads'])
#     if not exists:
#         st.session_state['chat_threads'].append(
#             {
#                 "id": thread_id,
#                 "title": "New Chat",
#                 "created_at": datetime.now()
#             }
#         )

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])

def generate_thread_title(first_message: str):
    prompt = f"""
You generate titles for chat conversations.

Rules:
- Maximum 5 words.
- No quotation marks.
- No punctuation.
- No prefixes like "Title:".
- Return only the title.

User's first message:
{first_message}
"""
    response = llm.invoke(prompt)
    return response.content.strip()

# def update_thread_title(thread_id, title):
#     for thread in st.session_state['chat_threads']:
#         if thread["id"] == thread_id:
#             thread["title"] = title
#             break


# **************************** Session Setup ****************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
    create_thread(st.session_state["thread_id"])

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

# add_thread(st.session_state['thread_id'])


# **************************** Sidebar UI ****************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread in st.session_state['chat_threads']:
    if st.sidebar.button(
        thread["title"],
        key=str(thread["id"])
    ):
        st.session_state['thread_id'] = thread["id"]
        messages = load_conversation(thread["id"])

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            elif isinstance(msg, AIMessage):
                role = 'assistant'
            else:
                continue
            temp_messages.append(
                {
                    'role': role,
                    'content': msg.content
                }
            )

        st.session_state['message_history'] = temp_messages

# **************************** Main UI ****************************

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}   # generates a dynamic thread_id instead of generating and naming a thread manually
user_input = st.chat_input('Type Here')

if user_input:

    st.session_state['message_history'].append({'role':'user', 'content':user_input})

    if len(st.session_state['message_history']) == 1:
        title = generate_thread_title(user_input)
        update_thread_title(st.session_state['thread_id'], title)
        st.session_state["chat_threads"] = retrieve_all_threads()

    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config = CONFIG,
                stream_mode="messages"
            )
        )

    st.session_state['message_history'].append({'role':'assistant', 'content':ai_message})
