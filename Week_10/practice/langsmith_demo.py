from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash"
)

prompt = ChatPromptTemplate.from_template("write a poem about {topic}")

chain = prompt | llm
response = chain.invoke({"topic": "Cat"})
print(response.content)


