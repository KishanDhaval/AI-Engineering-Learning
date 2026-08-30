import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from Week_4 import config
from Week_4.llm import cloud_llm
from Week_4.store import get_vectorstore

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an official HR Policy Assistant for Innvonix.
Answer the question using ONLY the provided HR policy context below.
If the answer is not mentioned in the context, say: "I am sorry, but that information is not available in the Innvonix HR Policy document."

Context:
{context}"""),
    ("human", "{query}")
])

def ask_cloud_bot(query: str):
    start = time.time()
    try:
        vectorstore = get_vectorstore(use_local=False)

        # Similarity search with FAISS scores
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=config.TOP_K)
        print(docs_and_scores)
        # Format Context
        context_blocks = [doc.page_content.strip() for doc, _ in docs_and_scores]
        context = "\n\n---\n\n".join(context_blocks)

        # Chain
        chain = PROMPT | cloud_llm | StrOutputParser()
        answer = chain.invoke({"context": context, "query": query}).strip()

        return answer, docs_and_scores, round(time.time() - start, 2)
    except Exception as e:
        return f"[ERROR] Cloud Bot Failed: {str(e)}", [], round(time.time() - start, 2)

print("\n=== CLOUD HR BOT (Mistral AI) ===")
while True:
    q = input("\nAsk Cloud Bot (or type 'quit'): ").strip()
    if not q: continue
    if q.lower() in ["quit", "exit"]: break

    print("Thinking...")
    answer, chunks, latency = ask_cloud_bot(q)

    print(f"\nLATENCY: {latency}s")
    print("-" * 60)
    print("RETRIEVED CHUNKS & FAISS SCORES:")
    for i, (doc, score) in enumerate(chunks, 1):
        page = doc.metadata.get("page", 0) + 1
        snippet = doc.page_content.replace("\n", " ")[:120]
        print(f"  Chunk #{i} | Page {page} | Score: {score:.4f}")
        print(f"  \"{snippet}...\"")

    print("-" * 60)
    print("RESPONSE:")
    print(answer)
    print("=" * 60)
