'''
- load doc
- chunk data
- embeding generation
- store to vector database
- new req 
- embed
- search 
- pass to llm (search result + original query)
'''

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# loading doc
loader = TextLoader("data.txt")
data = loader.load()

# splitting
spliter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
)
chunks = spliter.split_documents(data)

# embedding model
embeddingModel = MistralAIEmbeddings(
    model = "mistral-embed"
)

# vector store
vectorStore = FAISS.from_documents(
    chunks, 
    embeddingModel
)

print("\nvector store created")

# query
query = "Why man value banana in story?"

# search
result = vectorStore.similarity_search(query, k=3)

print("\n====== similarity search result ======")
for doc in result:
    print("\n--- Document --- ", doc.page_content)

# llm
llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.2 ,
    max_retries=2,
)

# prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the question based on the context provided. Answer only nothing else if not contains in context then just say i dont know about that."),
        ("human", "Question: {question}\nContext: {context}"),
    ]
)

# context
context = "\n\n".join([doc.page_content for doc in result])

chain = prompt | llm 

response = chain.invoke(
    {
        "question": query,
        "context": context,
    }
)

print("\n ***** final answer ***** \n",response.content)