from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field    
import os

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

class Person(BaseModel):
    name: str = Field(description="name of the person")
    age: int = Field(gt=18, description="age of the person")
    city: str = Field(description="city where the person lives")
    
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name,age and city of the fictional {place} person \n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | llm | parser

final_result = chain.invoke({'place':'India'})

print(final_result)