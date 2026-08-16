

from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import time,asyncio
load_dotenv()

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.2,
    max_tokens=1000,
    max_retries=2
)

question = HumanMessage(content="Explain the pythagoras theoram in 4 sentence")

start = time.perf_counter()

print("="*100)

res = llm.invoke([question])
print("invoke() time:", round(time.perf_counter() - start, 3))

print("="*100)

start = time.perf_counter()
first_chunk_time = None
for chunk in llm.stream([question]):
    if first_chunk_time == None:
        first_chunk_time = time.perf_counter()-start
    # print(chunk.content, end="")
print('time to first chunk: ', round(first_chunk_time,3), flush=True)

print("="*100)

async def run_concurrent():
    start = time.perf_counter()
    results = await asyncio.gather(
        llm.ainvoke("What is TCP?"),
        llm.ainvoke("What is UDP?"),
        llm.ainvoke("What is QUIC?"),
    )
    print("3 concurrent ainvoke() total time:", time.perf_counter() - start)
    return results

asyncio.run(run_concurrent())

async def run_astream():
    async for chunk in llm.astream(question):
        print(chunk.content, end="", flush=True)

asyncio.run(run_astream())