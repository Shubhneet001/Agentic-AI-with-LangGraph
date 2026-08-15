from langchain_groq import ChatGroq
from langchain_core.tools import tool
import requests
# from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv
from ddgs import DDGS         # DuckDuckGo Search
import os

load_dotenv()
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

# search_tool = DuckDuckGoSearchRun()      # old method


@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key={WEATHERSTACK_API_KEY}&query={city}'

  response = requests.get(url)

  return response.json()

@tool
def web_search(query: str) -> str:
    """Search the web for current information."""

    results = DDGS().text(
        query,
        max_results=5
    )

    return "\n\n".join(
        f"Title: {r['title']}\n"
        f"URL: {r['href']}\n"
        f"Content: {r['body']}"
        for r in results
    )

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Step 2: Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt

# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[web_search, get_weather_data],
    prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[web_search, get_weather_data],
    verbose=True,
    max_iterations=5
)

query1 = "What is the release date of Avengers Doomsday ?"
query2 = "What is the current temp of Delhi, India"
query3 = "List all the things that were announced in the Marvel's D23 event today"
query4 = "Identify the birthplace city of Kalpana Chawla (search) and give its current temperature."

# Step 5: Invoke
response = agent_executor.invoke({"input": query3})
print(response)

print(response['output'])
