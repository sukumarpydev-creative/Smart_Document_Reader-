
from langchain_classic.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
# from core.config import LLM_MODEL, TEMPERATURE


load_dotenv()
## defining tools

@tool
def mul (a:float, b:float) -> float:
    """Multiply two numbers."""
    return a * b

@tool
def add (a:float, b:float) -> float:
    """Addition of two numbers."""
    return a + b

tools = [mul, add ]

# Defining LLM
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

prom = [
    ("system", "You are a helpful Assistant"),
    ("human", "{input}"),
    ("assistant", "{agent_scratchpad}")
]

prompt = ChatPromptTemplate.from_messages(prom)
# Defining Agent and executor
agent = create_openai_functions_agent (llm = llm, tools = tools, prompt = prompt)

executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = "what is multiplication of 25 and 7"

print("Agent :", executor.invoke({"input": query})["output"])