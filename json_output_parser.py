from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

parser= JsonOutputParser()

# Prompt → get name, age and city of a fictional person
template1= PromptTemplate(
    template = "Give me 5 facts about {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
    
)

chain= template1 | llm | parser

result=chain.invoke({'topic': 'Albert Einstein'})

print(result)
print(type(result))