# Agentic AI with LangGraph

This repository contains everything I built while learning how to create AI agents with [LangGraph](https://langchain-ai.github.io/langgraph/). It starts with the basic building blocks, moves through the important features needed for real-world agents, and ends with a Streamlit chatbot that combines them all.

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Workflows](#1-workflows)
3. [Persistence](#2-persistence)
4. [Tools and MCP](#3-tools-and-mcp)
5. [Memory](#4-memory)
6. [RAG (Corrective and Self-RAG)](#5-rag-retrieval-augmented-generation)
7. [Subgraphs](#6-subgraphs)
8. [Streaming](#7-streaming)
9. [Human-in-the-loop](#8-human-in-the-loop)
10. [Observability with LangSmith](#9-observability-with-langsmith)
11. [Final Project: Streamlit Chatbot](#final-project-streamlit-chatbot)

---

## Core Concepts

A normal LLM call is a single question and a single answer. An **agent** is different: it can take several steps, make decisions, use tools, and repeat steps until a task is done. LangGraph helps you build this by describing the agent as a **graph**.

- **State**: the shared data that moves through the graph, such as the chat messages, retrieved documents, or any values you want to track. Every node can read it and update it.
- **Nodes**: normal Python functions. Each node does one job, like calling the LLM, running a tool, or checking an answer.
- **Edges**: the connections between nodes. They decide which node runs next.
- **Conditional edges**: edges that look at the state and choose the next node, which is how an agent makes decisions.

Think of it like a flowchart where each box is a function and the arrows are decided by your logic or by the LLM.

---

## 1. Workflows

Workflows are the different ways of arranging nodes in a graph. Choosing the right one depends on the task.

### Sequential
Steps run one after another, and each step uses the output of the previous one.

*Example:* take a topic → generate an outline → write a blog from the outline → write a title for the blog.

Use it when each step depends on the one before it.

### Parallel
Several independent steps run at the same time, and their results are combined at the end.

*Example:* evaluate an essay on clarity, grammar, and depth at the same time, then merge the three scores into a final score.

Use it when steps don't depend on each other. It saves time and keeps each task focused.

### Conditional
The graph picks a path depending on a condition in the state.

*Example:* classify a review as positive or negative. If it's positive, send a thank-you reply. If it's negative, run a deeper analysis and write a support response.

Use it when different inputs need different handling.

### Iterative
The graph loops back to an earlier step until a goal is reached or a limit is hit.

*Example:* generate a post → evaluate it → if it's not good enough, rewrite it using the feedback → evaluate again, up to a maximum number of tries.

Use it when quality improves with repeated attempts. A maximum iteration count is important so the loop can't run forever.

---

## 2. Persistence

By default, a graph forgets everything once a run ends. **Persistence** means saving the state of the graph at each step so it can be reloaded later. In LangGraph this is done with **checkpointers**, which save a snapshot (checkpoint) after every step.

### In-memory persistence
Checkpoints are stored in the program's RAM. It's the easiest to set up and great for testing, but everything is lost when the program stops.

### Database persistence
Checkpoints are stored in a database (for example SQLite or Postgres). The saved state survives restarts, so a user can come back days later and continue the same conversation. This is what you'd use in a real app.

### Threads
A **thread** is one separate conversation, identified by a `thread_id`. The checkpointer saves state per thread, so:
- Different users (or different chats) never mix their history
- Using the same `thread_id` again continues the old conversation
- Using a new `thread_id` starts a fresh one

### Fault tolerance
Because state is saved after every step, a crash doesn't mean starting over. If the program fails halfway (an API error, a network issue, or a server restart), the graph can resume from the last successful checkpoint. This saves time and API cost, especially for long multi-step workflows.

Persistence is also the foundation for memory, human-in-the-loop, and time travel (looking at or replaying past states), since all of them depend on saved state.

---

## 3. Tools and MCP

An LLM on its own can only generate text. **Tools** let it take actions and get real information, such as searching the web, doing calculations, or fetching data from an API.

### Tool calling
The LLM doesn't run the tool itself. The flow looks like this:

1. The LLM sees the user's question and the list of available tools
2. It decides whether a tool is needed and replies with the tool name and the inputs
3. LangGraph runs the tool in a tool node
4. The result goes back to the LLM
5. The LLM uses the result to write the final answer (or calls another tool)

This loop (LLM → tool → LLM) is what makes an agent "agentic", because the model decides what to do next.

### Creating custom tools
Any Python function can become a tool. You write the function, add a clear name, a description, and typed inputs, and the LLM uses that description to decide when to call it. A good description matters, since that is all the model knows about the tool.

### MCP (Model Context Protocol)
MCP is a standard way to connect AI apps to external tools and data sources. Instead of writing custom code for every service, you connect to an **MCP server** that exposes its tools in a common format, and your agent can discover and use them automatically.

Why it's useful:
- Reuse existing tool servers instead of building everything yourself
- Add or swap tools without changing the agent's core logic
- Works with many different tools through one standard interface

---

## 4. Memory

Memory is what makes an agent feel consistent and personal. There are two kinds.

### Short-term memory
Remembers things **within one conversation**. It's the message history of a thread, saved through the checkpointer. This is how the agent knows what "it" or "that" refers to when you say *"summarize that again"*.

Since LLMs have a limited context window, long chats can get too big. Common ways to handle this are trimming old messages or summarizing them.

### Long-term memory
Remembers things **across different conversations**. Instead of being tied to one thread, it's stored separately (in a store) and can be reused anywhere.

*Examples:* a user's name, preferences, goals, or facts they shared last week.

How it works in practice:
1. After or during a chat, important information is picked out and saved
2. In a new conversation, relevant memories are looked up
3. They are added to the prompt so the agent can use them

| | Short-term | Long-term |
|---|---|---|
| Scope | One thread | All threads |
| Stored as | Message history / checkpoints | Separate memory store |
| Good for | Following the current chat | Personalization over time |

---

## 5. RAG (Retrieval-Augmented Generation)

RAG lets an agent answer using **your own documents** instead of only what the LLM already knows. The basic flow is: split documents into chunks → convert them to embeddings → store them in a vector database → at question time, retrieve the most relevant chunks → give them to the LLM to write the answer.

Basic RAG has a weakness: if the retrieved chunks are wrong or irrelevant, the LLM may still produce a confident but incorrect answer. These two advanced patterns add checks to fix that.

### Corrective RAG (CRAG)
Adds a quality check **after retrieval**, before the answer is written.

1. Retrieve documents for the question
2. A grader (usually an LLM) checks whether each document is relevant
3. If the documents are good, use them to answer
4. If they're poor or missing, **correct** the situation, for example by rewriting the query or searching the web for better information
5. Generate the answer from the improved context

The idea: *don't trust retrieval blindly, check it and fix it.*

### Self-RAG
Adds checks **throughout the process**, including on the model's own answer. The agent reflects on its work at several points:

- Is retrieval even needed for this question?
- Are the retrieved documents relevant?
- Is the generated answer actually supported by the documents (no hallucination)?
- Does the answer actually address the user's question?

If a check fails, the agent loops back, for example by retrieving again or regenerating the answer.

The idea: *let the agent critique itself and retry until the answer is grounded and useful.*

| | Corrective RAG | Self-RAG |
|---|---|---|
| Main focus | Fixing bad retrieval | Checking retrieval and the answer |
| When it checks | After retrieval | At multiple stages |
| Typical fix | Rewrite query / web search | Re-retrieve or regenerate |

---

## 6. Subgraphs

A **subgraph** is a complete graph used as a single node inside another graph. As agents grow, one big graph becomes hard to read and maintain. Subgraphs let you split the work into smaller, reusable pieces. For example, one subgraph handles research, another handles writing, and a parent graph connects them.

Benefits: cleaner structure, easier testing of each part, and reuse in multiple places.

There are two ways a subgraph can relate to its parent's state:

### Shared state
The parent and subgraph use the **same state keys** (for example, both use `messages`). You can add the subgraph directly as a node, and data flows in and out automatically. It's simple and works well when both parts need the same information.

### Non-shared state
The subgraph has its **own, different state**. Because the schemas don't match, you write a small wrapper function that:
1. Converts the parent's state into the subgraph's input
2. Runs the subgraph
3. Converts the subgraph's output back into the parent's state

This keeps the subgraph independent and private, which is better when it's reusable or has its own internal data that the parent shouldn't see.

---

## 7. Streaming

Without streaming, the user waits until the entire response is ready. With **streaming**, output is sent piece by piece as it is produced, so the app feels fast and responsive.

LangGraph can stream different things:
- **Token streaming**: the LLM's reply word by word, like in ChatGPT
- **Step/update streaming**: progress after each node finishes (for example "retrieving documents… grading… answering…")
- **State streaming**: the full state after every step, useful for debugging

It matters most for long tasks and chat interfaces, where seeing progress is much better than staring at a blank screen.

---

## 8. Human-in-the-loop

Sometimes an agent shouldn't act fully on its own. **Human-in-the-loop (HITL)** lets the graph **pause** at a chosen point, wait for a person, and then **continue** based on their input.

Common uses:
- **Approval**: confirm before a risky action (sending an email, making a payment, deleting data)
- **Review and edit**: let a human correct the agent's draft or tool inputs
- **Providing missing information**: ask the user a question mid-run

How it works:
1. The graph reaches an interrupt point and stops
2. Its state is saved (this is why persistence is required)
3. A human reviews and responds (approve, reject, or edit)
4. The graph resumes from exactly where it stopped, using that response

Because the state is saved, the pause can last seconds or days without losing anything.

---

## 9. Observability with LangSmith

Agents make many decisions across many steps, so when something goes wrong it's hard to tell why. **LangSmith** records a **trace** of every run. A trace shows:

- Each node and LLM call in order
- The exact input and output of every step
- Which tools were called and what they returned
- Errors, time taken, and token usage (cost)

This helps you debug wrong answers, find slow or expensive steps, and understand how the agent reached its result. Setting it up mostly needs a LangSmith API key and a few environment variables, after which traces appear automatically.

---

## Final Project: Streamlit Chatbot

The `chatbot` folder contains a **Streamlit chatbot** that brings together the features learned in this repo, including persistence, tools, memory, streaming, and more, in a real chat interface.



Python · LangGraph · LangChain · LangSmith · Streamlit · MCP
