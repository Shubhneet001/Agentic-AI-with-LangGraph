from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM App'     # making a new project for langsmith to store this file's traces

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
model2 = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)
parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser
config = {
    'run_name': 'sequential_chain',
    'tags': ['llm_app', 'report_generation', 'summarization'],
    'metadata': {'model1': 'llama-3.3-70b-versatile', 'model2': 'llama-3.1-8b-instant', 'parser': 'StrOutputParser'}
}                        # for adding metadata and tags
result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)