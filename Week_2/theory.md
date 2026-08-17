# Week 2 — AI Engineering: LangChain Fundamentals
### Prompts, Models, Messages, Templates, Structured Outputs, Parsers, Runnables & Chains

---

## 1. Models — Groq and Mistral (and Gemini) Integrations

### What
A LangChain "Chat Model" is a Python class that wraps a provider's raw API (Groq, Mistral, Gemini, OpenAI, ...) behind one identical interface.

### Why
Without this wrapper, switching providers means rewriting your request-building code, your streaming code, your error handling — everything. With it, you change one constructor line and everything downstream (prompts, chains, parsers) keeps working unchanged. This is the entire value proposition of LangChain.

### How

```python
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_retries=2,
)

mistral_llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.7,
)

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
)

response = groq_llm.invoke("Explain what a race condition is in one sentence.")
print(response.content)
```

Notice: **the calling code (`.invoke(...)`) is identical across all three.** Only the constructor line changes. This means a finding like "Groq is fast but shallow, Gemini is slower but contextually sharper" is now a one-line swap away from being empirically re-verified inside a single script instead of three separate scripts.

API keys are picked up automatically from environment variables (`GROQ_API_KEY`, `MISTRAL_API_KEY`, `GOOGLE_API_KEY`) — you don't have to pass them manually unless you want to override.

### Model invocation methods — `invoke` / `stream` / `ainvoke` / `astream`

This is the part of "Models" you must know cold, because it directly extends sync vs. async APIs and SSE streaming.

**`invoke()` — synchronous, blocking, single result.**
Sends the whole prompt, waits, gets the whole response back at once. Simplest option; blocks your program until the full answer is generated.
```python
response = groq_llm.invoke("What is a deadlock?")
print(response.content)
```

**`stream()` — synchronous, but yields chunks as they arrive.**
Instead of waiting for the full answer, you get a generator that yields small pieces (tokens/chunks) as the model produces them — this is what powers a typewriter-style UI. This is the exact mechanism you'd plug into a FastAPI + SSE endpoint: each chunk from `.stream()` becomes one `data: ...` event sent to the browser.
```python
for chunk in groq_llm.stream("Explain event loops in 3 sentences."):
    print(chunk.content, end="", flush=True)
```

**`ainvoke()` — asynchronous, non-blocking, single result.**
Same as `invoke()`, but it's a coroutine — it doesn't block Python's event loop while waiting on the network. This matters when you're serving many concurrent users (like a FastAPI backend, or CreatorOS handling many simultaneous Socket.io connections): while one request is waiting on Groq's servers, the event loop is free to handle other requests instead of sitting idle.
```python
import asyncio

async def main():
    response = await groq_llm.ainvoke("What is idempotency?")
    print(response.content)

asyncio.run(main())
```

**`astream()` — asynchronous streaming.**
Combine both: non-blocking *and* chunked. This is what you'd use inside an `async def` FastAPI route that also streams via SSE — the ideal combination for a production LLM backend serving multiple users at once.
```python
async def main():
    async for chunk in groq_llm.astream("Explain backpressure in streaming systems."):
        print(chunk.content, end="", flush=True)

asyncio.run(main())
```

**Rule of thumb for when to use which:** `invoke()` for simple scripts and background jobs where nothing else is competing for attention. `stream()`/`astream()` whenever a human is watching a UI and perceived latency matters. `ainvoke()`/`astream()` specifically when your app is a server handling concurrent requests — your actual production scenario, given your FastAPI + Socket.io background.

---

## 2. Messages — SystemMessage, HumanMessage, AIMessage, ToolMessage

### What
Messages are the standardized version of the "role" system from raw Chat Completions calls (`system` / `user` / `assistant` / `tool`). LangChain wraps each role in its own Python class instead of a raw dictionary.

### Why
Type safety and IDE autocomplete aside, the real reason this matters: a conversation is just a Python **list** of these objects, in order, and that list *is* the model's entire memory of the conversation — nothing is remembered unless it's in that list. Understanding messages means understanding exactly what the model does and doesn't know at any point in a multi-turn exchange.

### How

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
```

**`SystemMessage`** — sets the model's behavior/persona for the whole conversation. Equivalent to the `system` role. Sent once, usually first.
```python
SystemMessage(content="You are a terse senior backend engineer. Answer in 2 sentences max.")
```

**`HumanMessage`** — what the user typed. Equivalent to `user` role.
```python
HumanMessage(content="Why would I choose SSE over WebSockets for an LLM streaming endpoint?")
```

**`AIMessage`** — what the model replied with. Equivalent to `assistant` role. You'll mostly *receive* these back from `.invoke()`, but you also *send* them back in when manually reconstructing multi-turn history (e.g., a chat app storing past turns so the model "remembers" earlier in the conversation).
```python
AIMessage(content="SSE is simpler over plain HTTP and one-directional, which matches how a token stream from an LLM actually behaves.")
```

**`ToolMessage`** — the result of executing a tool/function call the model requested. This is the LangChain equivalent of function calling: the model says "call `get_weather(city='Mumbai')`", your code actually runs that function, and you send the result back wrapped in a `ToolMessage` so the model can use it in its next reply. It requires a `tool_call_id` linking it back to the specific call the model made — without that ID the model can't tell which of possibly several tool calls this result answers.
```python
ToolMessage(content="28°C, humid", tool_call_id="call_abc123")
```

A full conversation is just a Python list of these:

```python
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="I love programming."),
]
response = groq_llm.invoke(messages)
print(response.content)   # -> "J'aime programmer."
```

To continue the conversation, you append the `AIMessage` you got back, then a new `HumanMessage`, and invoke again with the whole growing list — that's the entire mechanism behind "memory" in a chatbot.

**Shortcut syntax:** for quick scripts, LangChain also accepts plain tuples instead of message objects — useful, but less explicit:
```python
messages = [
    ("system", "You are a helpful translator."),
    ("human", "I love programming."),
]
```
Use the explicit classes in real application code (readability, autocomplete, and you'll need them anyway once `ToolMessage` enters the picture). Use tuples only for quick throwaway scripts.

---

## 3. Prompt Templates

### What
A `ChatPromptTemplate` is a reusable, parameterized version of a prompt — the same relationship an f-string has to a fixed string, except aware of message roles and able to validate its own inputs.

### Why
A prompt in a real application needs to be: reusable across many inputs without copy-pasting text, validated (did the caller forget a variable?), composed of multiple message roles (not just one blob of text), and defined once so every part of your codebase uses the exact same wording instead of subtly-different variants drifting apart over time. A plain f-string gives you none of this — it happily lets you forget a variable and send `"Review this  function:"` with a silent gap, and it has no concept of "this part is system instructions, this part is the user's input."

### How

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert code reviewer. Be specific and concise."),
    ("human", "Review this {language} function:\n\n{code}"),
])

formatted = prompt.invoke({"language": "Python", "code": "def add(a,b): return a+b"})
# formatted is a list of proper Message objects, ready to hand to a model
```

Notice `{language}` and `{code}` are placeholders — like f-string `{}` fields, but LangChain validates that you supplied every variable the template needs, and the template object itself is reusable and importable across your whole codebase.

`ChatPromptTemplate` is itself a **Runnable** (see §6) — meaning it has `.invoke()` just like a model does, and it can be piped with `|` directly into a model, which is exactly what a "chain" is (§7).

**`.partial()`** lets you "pre-fill" some variables now and leave the rest for later — useful when one part of a template (like formatting instructions from a parser) is known ahead of time but the user's actual input isn't yet.
```python
prompt = prompt.partial(language="Python")
```

**`MessagesPlaceholder`** is used when you need to inject a variable-length chat history into a fixed template (essential for multi-turn chatbots — this is how message lists from §2 slot into a template):
```python
from langchain_core.prompts import MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
```

---

## 4. Output Parsers — With and Without `with_structured_output`

This is the part of the pipeline that turns the model's raw reply into data your code can actually use — a dict, a list, a validated object — instead of a wall of text you'd have to regex apart by hand.

### 4.1 The schema underneath it all — Pydantic

Before parsing anything, you need to define *what correct output looks like*. Pydantic is a Python library for data contracts — a class describing exactly what fields a piece of data must have and what type each one is. Think of it as a bouncer at a door: data either matches the shape you declared, or it's rejected with a clear error before it's ever used downstream.

```python
from pydantic import BaseModel, Field
from typing import List

class ReviewAnalysis(BaseModel):
    sentiment: str = Field(description="One of: positive, negative, neutral")
    key_issues: List[str] = Field(description="Specific problems mentioned in the review, if any")
    summary: str = Field(description="One-sentence summary of the review")
```

Two things matter enormously here, and both are easy to miss:

1. **The `description` in `Field(...)` isn't just documentation for humans.** LangChain converts this class into a JSON Schema and sends that schema — descriptions included — to the LLM as part of the instructions for what to generate. A vague description produces vague or wrong output. A precise description ("One of: positive, negative, neutral" rather than just "the sentiment") measurably improves accuracy, because you're literally telling the model what values are valid.
2. **Types are enforced, not suggested.** If the model tries to return `"key_issues": "shipping was late"` (a string) instead of a list, Pydantic raises a validation error rather than silently accepting malformed data. This is your first line of defense against a confidently wrong *type* — not just a wrong fact — getting caught mechanically instead of slipping into production.

### 4.2 WITHOUT `with_structured_output` — manual Output Parsers

This is the older, more manual approach, and understanding it matters because it shows you exactly what's happening under the hood, and it's still the right tool when a provider doesn't support tool-calling well.

**How it works, step by step:**
1. You create a parser from your Pydantic schema.
2. You ask the parser for `format_instructions` — plain-English text describing the exact JSON shape the model should reply with — and inject that text into your prompt.
3. The model replies with plain text (hopefully valid JSON).
4. You call `.parse()` yourself on that raw text, which either succeeds and gives you a Pydantic object, or raises an error if the text doesn't match.

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Analyze the review. {format_instructions}"),
    ("human", "{review_text}"),
]).partial(format_instructions=parser.get_format_instructions())

raw_response = groq_llm.invoke(prompt.invoke({"review_text": "Late delivery but great product."}))
result = parser.parse(raw_response.content)   # -> ReviewAnalysis(...) or raises an error
```

**The catch:** because this parser works on raw *text*, nothing forces the model to actually follow the format instructions. It can wrap the JSON in extra commentary, forget a field, or produce malformed JSON — and `.parse()` will throw. This failure mode is exactly why retry/backoff logic exists. LangChain has `OutputFixingParser` and `RetryOutputParser`, which wrap a base parser and, on failure, send the broken output *back* to an LLM with instructions to fix it — an automated version of manual retry logic.

There's also a simpler, unvalidated option: **`StrOutputParser`** — doesn't validate anything, just extracts `.content` from the `AIMessage` so you get a plain string instead of a message object. Useful mainly as the final step in a chain when you don't need structure at all.
```python
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
parser.invoke(AIMessage(content="hello"))   # -> "hello"
```

And **`JsonOutputParser`** — a looser cousin that parses whatever JSON the model produced into a plain Python dict, without requiring a full Pydantic round-trip.

### 4.3 WITH `with_structured_output` — the modern, preferred approach

**What it is:** `.with_structured_output(YourSchema)` wraps a model so that instead of returning free text you have to parse yourself, it returns an **already-parsed, already-validated instance of your Pydantic class directly**.

**Why it's better:** it doesn't rely on the model "choosing" to follow format instructions embedded in a prompt. Instead, LangChain converts your Pydantic schema into a **tool definition** — the exact same mechanism as function/tool calling — and asks the model to "call" that tool with the extracted data as arguments. This happens at the API level (via the provider's tool-calling feature), so it's enforced far more reliably than hoping free text happens to parse correctly.

```python
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

class AnswerWithJustification(BaseModel):
    """An answer to the user's question, with justification."""
    answer: str
    justification: str = Field(description="Why this answer is correct")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
structured_llm = llm.with_structured_output(AnswerWithJustification)

result = structured_llm.invoke("What weighs more, a pound of bricks or a pound of feathers?")
print(result.answer)          # "They weigh the same"
print(result.justification)   # "..."
print(type(result))           # <class '__main__.AnswerWithJustification'> — a real Python object, not a string
```

No `format_instructions` in the prompt, no manual `.parse()` call — the wrapped model just hands you the object.

**Method variants** — not every provider/model supports every one equally well, worth testing across Groq/Mistral/Gemini:
- `method="function_calling"` (default) — uses tool-calling under the hood, as described above.
- `method="json_schema"` — native structured-output enforcement at the API level, where supported.
- `method="json_mode"` — looser, just asks for valid JSON without a strict schema contract.

**`include_raw=True`** gets you both the parsed object *and* the original raw `AIMessage` — useful for debugging when the model's structured output doesn't match what you expected, or when you want to inspect exactly what the model produced before parsing.
```python
structured_llm = llm.with_structured_output(AnswerWithJustification, include_raw=True)
result = structured_llm.invoke("...")
# result = {"raw": AIMessage(...), "parsed": AnswerWithJustification(...), "parsing_error": None}
```

### 4.4 Which one to actually use

| | `PydanticOutputParser` (manual) | `.with_structured_output()` (modern) |
|---|---|---|
| Where enforcement happens | Nowhere — hopes the model follows text instructions | At the API level, via tool-calling |
| Failure mode | `.parse()` throws on malformed text | Rarer; can still fail if provider's tool-calling is weak |
| When to use | Provider doesn't support tool-calling well; you need to understand the mechanism manually | Default choice whenever the provider supports it |
| What you get back | Whatever you build in `.parse()` — object or a raised exception | Validated Pydantic object directly |

Prefer `.with_structured_output()` by default. Use manual parsers when you need to reproduce the mechanism yourself or work around a provider limitation — and always compare both, because *how* each one fails is as informative as whether it succeeds.

---

## 5. Runnables — Why, What, and How

### Why
Every piece covered so far — models, prompt templates, parsers — was built separately, by different people, for different purposes. Without a shared interface, combining them would mean writing custom glue code every time: "take the output of this prompt object, manually format it for this model object, manually pass the result to this parser object." That glue code would have to be rewritten for every new combination.

### What
A **Runnable** is a shared interface. Almost everything in LangChain — models, prompt templates, parsers, even plain Python functions — implements it. That means every one of them has `.invoke()`, `.batch()`, `.stream()`, `.ainvoke()`, `.abatch()`, `.astream()` — regardless of whether it's a model, a prompt template, or a parser. Once something is a Runnable, it can be connected to any other Runnable using the exact same syntax, with no custom glue code required.

This is the "same shape everywhere" idea from §0 (which described it across *providers*), taken one level deeper: the same idea applied across *every kind* of component in the pipeline, not just models.

### How

Three Runnables worth knowing by name, beyond models/prompts/parsers (which are already Runnables on their own):

**`RunnableLambda`** — wraps any plain Python function so it behaves like a Runnable and can be inserted into a chain. Use this whenever you need a custom transformation step (cleaning text, reshaping a dict) that isn't itself a model or parser.
```python
from langchain_core.runnables import RunnableLambda

uppercase = RunnableLambda(lambda text: text.upper())
uppercase.invoke("hello")   # -> "HELLO"
```

**`RunnablePassthrough`** — forwards its input unchanged. Sounds trivial, but it's essential inside `RunnableParallel` when you need the original input to survive alongside a transformed value (e.g., keep the original review text *and* attach the LLM's analysis of it, instead of losing the original once it's been transformed).
```python
from langchain_core.runnables import RunnablePassthrough
```

**`RunnableParallel`** — runs multiple Runnables concurrently on the same input and combines their outputs into a dict. Useful for running several independent analyses on one input at once instead of sequentially waiting for each.
```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    original=RunnablePassthrough(),
    length=RunnableLambda(lambda x: len(x)),
)
parallel.invoke("hello world")
# -> {"original": "hello world", "length": 11}
```

Think of Runnables as LEGO bricks: individually simple, uniformly shaped, and designed to snap together in any combination — which is exactly what makes chains possible.

---

## 6. Chains / LCEL (LangChain Expression Language)

### What
A **chain** is multiple Runnables connected so the output of one becomes the input of the next. LCEL is the syntax for wiring them together using the pipe operator `|` — the same symbol as a Unix shell pipe, and it means the same thing: data flows left to right.

### Why
Without chains, you'd manually call `.invoke()` on the prompt, take its output, manually call `.invoke()` on the model with that output, take *that* output, manually call `.parse()` on it — three separate calls, three separate variables, and you'd have to rewrite all of it if you wanted `.stream()` or `.batch()` behavior instead. A chain collapses this into one object that behaves consistently no matter which mode you call it in.

### How

```python
chain = prompt | llm | parser
```

Read this exactly like a Unix pipeline: build the prompt from input variables → feed the formatted prompt to the model → feed the model's output into the parser. Each stage's output becomes the next stage's input automatically.

```python
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the review in one sentence."),
    ("human", "{review_text}"),
])

chain = prompt | groq_llm | StrOutputParser()

result = chain.invoke({"review_text": "The product broke after two days, very disappointed."})
print(result)   # -> plain string summary
```

Because `chain` is *itself* a Runnable (composition of Runnables is still a Runnable — that's the whole trick), you automatically get `.invoke()`, `.batch()` (run many inputs, some providers parallelize this), `.stream()`, and the async equivalents on the **whole pipeline**, for free, without writing any extra glue code:

```python
results = chain.batch([
    {"review_text": "Review 1..."},
    {"review_text": "Review 2..."},
])

for chunk in chain.stream({"review_text": "Review 3..."}):
    print(chunk, end="")
```

Under the hood, `prompt | llm | parser` constructs a `RunnableSequence` — an ordered list of Runnables that LangChain calls one after another, passing output to input at each step. This is also the mental model for debugging a broken chain: reasoning about it as "step 1's output must match what step 2 expects as input" is usually enough to find exactly where a pipeline is failing.

**Chains can also branch and merge**, by combining LCEL with `RunnableParallel` from §5 — e.g., running the same input through two different prompts/models at once and merging the results into a single dict before a final step processes both together.

---

## Quick-Reference Cheat Sheet

| Concept | One-line meaning |
|---|---|
| `ChatGroq` / `ChatMistralAI` / `ChatGoogleGenerativeAI` | Provider-specific class, identical interface |
| `SystemMessage` / `HumanMessage` / `AIMessage` / `ToolMessage` | Standardized conversation roles |
| `.invoke()` | Sync, blocking, one full response |
| `.stream()` | Sync, chunked response as it's generated |
| `.ainvoke()` | Async, one full response, doesn't block the event loop |
| `.astream()` | Async, chunked — best for production streaming servers |
| `ChatPromptTemplate` | Reusable, validated, parameterized prompt |
| `pydantic.BaseModel` | Data contract / schema definition |
| `PydanticOutputParser` (without structured output) | Manually parses raw text into a Pydantic object; can fail on malformed text |
| `.with_structured_output(Schema)` | Model returns a validated Pydantic object directly, enforced via tool-calling |
| `StrOutputParser` | Extracts plain string content, no validation |
| `RunnableLambda` | Wraps a plain function as a Runnable |
| `RunnablePassthrough` | Forwards input unchanged |
| `RunnableParallel` | Runs multiple Runnables concurrently, merges results into a dict |
| `prompt \| llm \| parser` | LCEL chain — a pipeline that's itself a Runnable |