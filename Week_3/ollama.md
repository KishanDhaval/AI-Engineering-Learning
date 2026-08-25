# Ollama & Local LLMs — Reference Guide

> Open source vs. proprietary models, Ollama internals, LangChain integration, and cloud models.

---

## 1. Open Source vs. Proprietary Models

### Open Source (Open-Weight) Models
The model's trained weights are published and downloadable. You run them yourself — no API call to a company's servers.

- **Examples**: Llama 3.3, Mistral, Gemma, DeepSeek, Qwen
- **How you use them**: download weights, run locally (Ollama) or self-host on a rented GPU
- **Cost**: free to use — you pay in compute/hardware instead
- **Data privacy**: nothing leaves your machine
- **Caveat**: many "open source" LLM licenses (e.g. Llama's) carry usage restrictions, so the term "open-weight" is technically more accurate than "open source"

### Proprietary Models
The company keeps weights private. You interact only through their API.

- **Examples**: GPT-4/5 (OpenAI), Claude (Anthropic), Gemini (Google)
- **How you use them**: API calls, pay-per-token
- **Cost**: usage-based, zero hardware needed
- **Data privacy**: prompts go to a third-party server
- **Advantage**: typically the most capable models, no infra management, always current

### Where Groq Fits (Hybrid Case)
Groq is a **proprietary inference service** (custom LPU chips) hosting an **open-weight model** (`llama-3.3-70b-versatile` from Meta). Groq itself isn't a model — it's fast, cloud-hosted infrastructure serving an open-weight model.

### Node.js Mental Model
| Concept | Node.js Analogy |
|---|---|
| Open-weight model | `git clone`-ing a package and running it yourself |
| Proprietary model | A SaaS API (Stripe, Twilio) — you never see internals |

---

## 2. What Is Ollama?

Ollama is a **runtime** that lets you download and run open-weight LLMs directly on your own machine — no cloud infra, no API keys.

**What it does under the hood:**
1. **Packages models** — pulls open-weight models (often quantized/compressed) into an easy-to-run format
2. **Runs a local inference server** — `ollama serve` starts an HTTP server at `localhost:11434`
3. **Manages model lifecycle** — pulling, loading into memory/GPU, switching, unloading (like a package manager, but for models)

**The stack:**
```
Llama 3.3 (open weights, from Meta)
        ↓
   Ollama (runtime that loads and serves it)
        ↓
  localhost:11434 (local API endpoint)
        ↓
   Your Python/Node code talks to it
```

### Node.js Analogy
`ollama pull llama3.3` ≈ `npm install`. `ollama run llama3.3` ≈ `node server.js`, spinning up a local server you hit with HTTP requests.

---

## 3. The Five Ways to Interact With Ollama

| Layer | What It Is | Best For |
|---|---|---|
| **CLI** (`ollama run`) | Direct terminal chat | Quick manual testing |
| **REST API** | Raw HTTP to `localhost:11434` | No dependencies, full control |
| **Official Python lib** (`ollama`) | Thin wrapper over REST | Clean sync/async Python code |
| **OpenAI-compatible endpoint** | `openai` SDK pointed at `localhost:11434/v1` | Reusing existing OpenAI-based code |
| **LangChain** (`ChatOllama`) | Framework abstraction | Chains, easy provider swapping |

### 3.1 CLI
```bash
ollama run llama3.3                          # interactive chat
ollama run llama3.3 "Explain recursion"       # one-shot prompt
ollama pull llama3.3                          # download only
ollama list                                   # list downloaded models
ollama rm llama3.3                            # remove a model
```

### 3.2 REST API
```python
import requests

response = requests.post("http://localhost:11434/api/generate", json={
    "model": "llama3.3",
    "prompt": "What is a REST API?",
    "stream": False
})
print(response.json()["response"])
```

**Core endpoints:**

| Endpoint | Purpose |
|---|---|
| `POST /api/generate` | Single-prompt completion (no conversation state) |
| `POST /api/chat` | Multi-turn chat with message roles |
| `POST /api/embeddings` | Generate vector embeddings |
| `GET /api/tags` | List locally available models |
| `POST /api/pull` | Download a model (streams progress) |
| `DELETE /api/delete` | Remove a model |
| `POST /api/show` | Get model details (Modelfile, params, template) |
| `POST /api/copy` | Duplicate a model under a new name |
| `POST /api/create` | Build a custom model from a Modelfile |

**Common `options` parameters:**
```python
"options": {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40,
    "num_ctx": 4096,        # context window size
    "num_predict": 256,     # max tokens to generate
    "repeat_penalty": 1.1,
    "seed": 42               # reproducibility
}
```

**Streaming** — Ollama returns **newline-delimited JSON** (one JSON object per line), unlike Groq/OpenAI's SSE (`data: {...}`) format:
```python
response = requests.post("http://localhost:11434/api/chat", json={
    "model": "llama3.3",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": True
}, stream=True)

for line in response.iter_lines():
    if line:
        chunk = json.loads(line)
        print(chunk["message"]["content"], end="", flush=True)
```

### 3.3 Official Python Library
```bash
pip install ollama --break-system-packages
```
```python
import ollama

response = ollama.chat(model="llama3.3", messages=[
    {"role": "user", "content": "Explain async/await simply"}
])
print(response["message"]["content"])

# Streaming
for chunk in ollama.chat(model="llama3.3", messages=[...], stream=True):
    print(chunk["message"]["content"], end="", flush=True)

# Embeddings
result = ollama.embeddings(model="llama3.3", prompt="Hello world")
```

**Async client** (relevant to `asyncio.gather` patterns from Week 2):
```python
from ollama import AsyncClient
response = await AsyncClient().chat(model="llama3.3", messages=[...])
```

### 3.4 OpenAI-Compatible Endpoint
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # key ignored
response = client.chat.completions.create(model="llama3.3", messages=[...])
```
Useful for reusing existing OpenAI-based code with zero rewrite.

### 3.5 LangChain
```bash
uv add langchain-ollama
```
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.3", temperature=0, base_url="http://localhost:11434")
response = llm.invoke("What is dependency injection?")
```

---

## 4. How LangChain "Knows" Where to Route

**Key concept: there is no automatic detection.** Routing is determined entirely by **which class you instantiate** — not by any runtime logic.

```python
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

groq_llm = ChatGroq(model="llama-3.3-70b-versatile")   # → api.groq.com
local_llm = ChatOllama(model="llama3.3")                # → localhost:11434
```

Each class has a default (or overridable) `base_url` baked in:

| Class | Default Target |
|---|---|
| `ChatGroq` | `https://api.groq.com/openai/v1` |
| `ChatOpenAI` | `https://api.openai.com/v1` |
| `ChatOllama` | `http://localhost:11434` |
| `ChatAnthropic` | `https://api.anthropic.com` |

**What "provider-agnostic" actually means:** switching providers = changing the class instantiated. Everything downstream (prompts, chains, parsers) stays identical because all classes implement LangChain's common `BaseChatModel` interface.

```python
prompt = ChatPromptTemplate.from_template("Explain {topic}")
llm = ChatGroq(model="llama-3.3-70b-versatile")   # or ChatOllama(...)
chain = prompt | llm
```

**Dynamic routing** (e.g., prefer local, fall back to cloud) must be written explicitly — it is not a LangChain feature:
```python
def get_llm(prefer_local=True):
    if prefer_local:
        try:
            requests.get("http://localhost:11434", timeout=1)
            return ChatOllama(model="llama3.3")
        except requests.exceptions.ConnectionError:
            pass
    return ChatGroq(model="llama-3.3-70b-versatile")
```

### Node.js Analogy
Like instantiating two separate SDK clients (`new Stripe(...)`, `new Twilio(...)`) — both expose similarly shaped methods, but the destination was fixed at construction time, not at call time.

---

## 5. Structured Output (Pydantic) with Local Models

```python
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

class Summary(BaseModel):
    title: str = Field(description="A short title")
    key_points: list[str] = Field(description="3 bullet points")

llm = ChatOllama(model="llama3.3", temperature=0)
structured_llm = llm.with_structured_output(Summary)
result = structured_llm.invoke("Summarize how REST APIs work")
```

⚠️ **Reliability gap**: `with_structured_output` is noticeably weaker on local/quantized models than on cloud models (Groq/OpenAI). Cloud models are specifically fine-tuned for tight schema/function-calling adherence; local models — especially smaller ones — can drift or return malformed JSON. Worth quantifying directly in any local-vs-cloud comparison.

---

## 6. Async & Concurrency Nuance

```python
import asyncio
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.3")

async def run_all():
    tasks = [llm.ainvoke(q) for q in ["What is TCP?", "What is UDP?", "What is HTTP?"]]
    results = await asyncio.gather(*tasks)

asyncio.run(run_all())
```

**Important distinction from Groq**: `asyncio.gather` helps a lot with Groq because it's I/O-bound (waiting on network). With local Ollama, you're bound by your machine's **compute**, not network wait — so concurrent requests may still process sequentially on your GPU/CPU unless `OLLAMA_NUM_PARALLEL` is configured. This is a good empirical thing to test rather than assume.

---

## 7. Tool Calling (Function Calling)

The model requests that *you* execute a function; you run it and feed the result back.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }
}]

response = requests.post("http://localhost:11434/api/chat", json={
    "model": "llama3.3",
    "messages": [{"role": "user", "content": "What's the weather in Vadodara?"}],
    "tools": tools,
    "stream": False
})

tool_calls = response.json()["message"].get("tool_calls", [])
# Execute the function yourself, then send the result back as a "tool" role message
```

**LangChain abstraction over the same mechanism:**
```python
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"22°C and sunny in {city}"

agent = create_tool_calling_agent(llm, [get_weather], prompt)
executor = AgentExecutor(agent=agent, tools=[get_weather], verbose=True)
```

**Reliability by model size**: `qwen3-coder`, `llama3.3`, `mistral`, `hermes3` handle tool calling reasonably. Models under ~8B parameters are unreliable — frequent malformed calls or hallucinated function names.

### Node.js Analogy
Same shape as a webhook handshake — the model says "POST to `/weather?city=X`", you make the call, then report the result back.

---

## 8. Modelfile

Ollama's equivalent of a **Dockerfile** — a declarative recipe for building a customized model on top of a base model (system prompt, default params, etc.) without repeating config in every API call.

```dockerfile
# Modelfile
FROM llama3.3

SYSTEM """
You are a senior backend engineer who explains concepts using Node.js/Express analogies.
Keep answers concise and code-first.
"""

PARAMETER temperature 0.3
PARAMETER num_ctx 8192
PARAMETER top_p 0.9
```

```bash
ollama create backend-tutor -f ./Modelfile
ollama run backend-tutor
```

```python
llm = ChatOllama(model="backend-tutor")  # system prompt already baked in
```

**Other directives:**
- `TEMPLATE` — override the chat prompt template
- `ADAPTER` — attach a LoRA fine-tune
- `LICENSE` — embed license text
- `MESSAGE` — seed few-shot conversation examples into the model itself

### Node.js Analogy
Like an Express app factory — `createApp({ cors: true, logger: 'combined' })` — vs. wiring middleware manually every time.

---

## 9. Orchestration with LangChain

### a) Conditional multi-provider routing
```python
def choose_model(input_dict):
    if len(input_dict["question"]) < 50:
        return ChatOllama(model="llama3.3")
    return ChatGroq(model="llama-3.3-70b-versatile")
```

### b) Fully local RAG pipeline
```python
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.3", temperature=0)
retriever = FAISS.load_local("my_index", embeddings, allow_dangerous_deserialization=True).as_retriever(k=3)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
```
Embeddings, retrieval, and generation all stay on-machine — zero data leaves.

### c) Agents with tool calling
See Section 7 — LangChain's `@tool` decorator auto-converts Python functions into the JSON schema Ollama expects.

---

## 10. Ollama Cloud (Large Models)

Lets you run models too large for local hardware (70B–480B+ params) by routing through Ollama's own servers — while keeping an **identical local API surface**.

```bash
ollama pull gpt-oss:120b-cloud   # downloads a tiny manifest only, not the model
ollama run gpt-oss:120b-cloud    # requests proxied to Ollama's infrastructure
```

**Mechanism**: your code still sends requests to `localhost:11434` exactly as normal. The local daemon detects the `:cloud` suffix, attaches your `ollama.com` auth credentials, and proxies the request to Ollama's remote infrastructure. Response streams back the same way.

```python
llm = ChatOllama(model="gpt-oss:120b-cloud")  # identical code shape to local models
```

### Comparison Table

| | Local (`llama3.3`) | Ollama Cloud (`gpt-oss:120b-cloud`) | Groq (`llama-3.3-70b-versatile`) |
|---|---|---|---|
| Runs on | Your machine | Ollama's servers | Groq's LPU chips |
| Size limit | Bound by your VRAM/RAM | None (up to 480B+) | Whatever Groq hosts |
| Cost | Free (your hardware) | Ollama Pro subscription | Pay-per-token |
| Privacy | Fully private | Leaves your machine | Leaves your machine |
| Maturity | Stable | Actively reported reliability issues (high failure rates on some cloud models, as of mid-2026) | Generally solid |

**Practical note**: Ollama Cloud is a genuinely separate product from Groq, despite both serving Llama-family models. Given current reliability concerns, it's best treated as experimental — cleaner to compare plain local Ollama vs. Groq for core benchmarking work.

---

## 11. Full Mental Model Summary

| Ollama Feature | Node.js / Backend Equivalent |
|---|---|
| REST API | Raw Express routes hit with `fetch` |
| Official Python/JS lib | Official SDK wrapping raw HTTP |
| OpenAI-compatible endpoint | Adapter pattern for drop-in compatibility |
| Tool calling | Webhook-style callback (model requests, you execute, you report back) |
| Modelfile | `createApp(config)` factory baking in middleware/config |
| LangChain (`ChatOllama`) | Dependency injection — swappable provider modules behind one interface |
| Cloud models (`:cloud`) | Local server acting as an authenticated reverse proxy to a bigger upstream service |

---

## 12. Key Takeaways for Exercise 3 (Ollama vs. Groq)

1. **Routing is explicit, not automatic** — `ChatOllama` vs `ChatGroq` is a class choice, not runtime detection.
2. **Structured output is less reliable locally** — worth measuring directly with Pydantic schemas.
3. **Async concurrency helps less locally** — Groq benefits from `asyncio.gather` (I/O-bound); local Ollama is compute-bound and may not.
4. **Missing local models fail loudly** — no silent fallback to any API unless you explicitly use a `:cloud` tag.
5. **Ollama Cloud ≠ Groq** — separate products; Ollama Cloud is currently less mature and better avoided for core benchmarking.
6. **Latency, quality, and reliability** are the three axes to measure and document in the comparison writeup.