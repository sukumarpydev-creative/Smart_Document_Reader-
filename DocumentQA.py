from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
# from langchain.chains import LLMChain

# we need to create Prompt, LLM, LLM Chain

# Prompt Template creation
prompt = PromptTemplate.from_template("Introduce your name, and Summarize the points into JSON format  :\n\n{text}")

# Setting up the model
llm = OllamaLLM(model='Jimmy')

# Parsing in JSON, to get output in Pure JSON format
parser = JsonOutputParser()

# Setting up the chain
llmchain = prompt | llm | parser

# Input Text
in_text = """
Sukumar is Good Boy. He is calm, silen. He likes to code. He learned Python, Javascript, and JavaScript, 
REactJS, HTML, CSS. Now he is learning LangChain. He want to build LLM Application.
"""

result = llmchain.invoke({'text': in_text})
print(result)


