import streamlit as st
from backend import (
    chatbot,
    create_thread,
    llm,
    retrieve_all_threads,
    update_thread_title,
    ingest_pdf
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


# **************************** Session Setup ****************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
    create_thread(st.session_state["thread_id"])

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}


# **************************** Sidebar UI ****************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.divider()

# --- PDF Upload / RAG Section ---
st.sidebar.header('Document Upload (RAG)')
uploaded_file = st.sidebar.file_uploader("Upload a PDF to chat with", type=["pdf"])

if uploaded_file is not None:
    if st.sidebar.button("Process PDF"):
        with st.spinner("Ingesting document..."):
            try:
                file_bytes = uploaded_file.read()
                result = ingest_pdf(
                    file_bytes=file_bytes, 
                    thread_id=st.session_state["thread_id"], 
                    filename=uploaded_file.name
                )
                st.session_state["ingested_docs"][st.session_state["thread_id"]] = result
                st.sidebar.success("PDF processed successfully!")
            except Exception as e:
                st.sidebar.error(f"Error processing PDF: {str(e)}")

# Display active document for the current thread if it exists
current_doc = st.session_state["ingested_docs"].get(st.session_state["thread_id"])
if current_doc:
    st.sidebar.info(f"📄 **Active Document:** {current_doc['filename']}\n\n"
                    f"*{current_doc['documents']} pages, {current_doc['chunks']} chunks*")

st.sidebar.divider()

# --- Chat History Section ---
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

CONFIG = {
    "configurable": {"thread_id": st.session_state["thread_id"]},
    "metadata": {
        "thread_id": st.session_state["thread_id"]
    },
    "run_name": "chat_turn"
}

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
        def ai_only_stream():
            used_tools = set()
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    # AI messages contain tools calls when the model requests a tool
                    for tool_call in message_chunk.tool_calls:
                        used_tools.add(tool_call["name"])
                    # yield only ai messages
                    yield message_chunk.content 

            if used_tools:
                st.caption("Tools Used: " + ", ".join(sorted(used_tools)))

        ai_message = st.write_stream(ai_only_stream())

    st.session_state['message_history'].append({'role':'assistant', 'content':ai_message})