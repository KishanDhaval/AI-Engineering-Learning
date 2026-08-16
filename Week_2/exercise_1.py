'''
    Todo : 
    making reusable prompt with chat prompt templete
'''

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=1000,
    max_retries=2
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a senior {language} engineer doing a strict code review.\n Point out bugs, bad naming, and missing error handling."),
    ("human", "Review this code : \n\n {code}")
])

chain = prompt | llm 

res = chain.invoke({
    "language": "Python",
    "code": "def add(a, b): return a+b"
})

print(res.content)
