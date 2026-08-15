# Week 1 — AI Engineering Fundamentals

*Reference notes covering the foundational concepts of AI Engineering.*

---

## Table of Contents

1. [What is AI Engineering](#1-what-is-ai-engineering)
2. [AI Engineering vs. Agentic AI vs. Generative AI vs. AI Agents](#2-ai-engineering-vs-agentic-ai-vs-generative-ai-vs-ai-agents)
3. [Tokens and Tokenizers in Detail](#3-tokens-and-tokenizers-in-detail)
4. [Temperature, Top-p, Context Window, Max Tokens, Hallucination](#4-temperature-top-p-context-window-max-tokens-hallucination)
5. [Transformer Architecture and Attention](#5-transformer-architecture-and-attention)
6. [Why LangChain?](#6-why-langchain)

---

## 1. What is AI Engineering

**AI Engineering** is the discipline of building applications on top of *existing* foundation models (GPT, Claude, Gemini, Llama, etc.) rather than training models from scratch. It is distinct from traditional **Machine Learning Engineering**, which focuses on training and fine-tuning models from the ground up.

### Core focus areas

- **Prompt engineering** — designing inputs that reliably get the model to do what you want
- **Retrieval-Augmented Generation (RAG)** — connecting models to external knowledge (databases, documents, APIs)
- **Evaluation** — measuring whether outputs are actually good (no simple accuracy metric exists for generative output)
- **Agents & tool use** — giving models the ability to call functions, browse, or take multi-step actions
- **Fine-tuning (sometimes)** — light adaptation of an existing model, not training one from zero
- **Guardrails & safety** — filtering, moderation, handling hallucination
- **Systems/infrastructure** — latency, cost, caching, orchestration, pipeline design

### Traditional ML Engineering vs. AI Engineering

| Traditional ML Engineering | AI Engineering |
|---|---|
| Trains models from raw data | Builds on top of pre-trained foundation models |
| Requires large datasets and compute | Requires far less data/compute per project |
| Deep focus on model architecture | Deep focus on product behavior, prompting, evaluation |
| Slower iteration (training cycles) | Fast iteration (prompt/config changes) |
| ML-specific skills (statistics, model theory) | More software-engineering-centric skills |

### Why prompt engineering alone is not enough

- **Doesn't scale reliability** — a prompt that works on 20 hand-picked examples can silently fail on the long tail of real inputs
- **Doesn't fill knowledge gaps** — no wording fixes a model not having the right data (that requires RAG)
- **Doesn't guarantee safety** — you can *ask* a model not to leak data; you can't *guarantee* it without external validation
- **Doesn't manage multi-step tasks** — complex work needs orchestration, not clever phrasing
- **Doesn't provide observability** — you can't fix what you don't measure (logging, tracing, eval pipelines)
- **Is brittle across models/versions** — a prompt tuned for one model may break on the next

> **One-line summary:** AI Engineering is less "build a brain" and more "build a great product using a brain someone else already built."

### The three-layer mental model

```
Model      → capability, but unreliable and stateless
Context    → what it knows right now (RAG, memory, prompts)
System     → what makes it safe, reliable, fast, cheap
Evaluation → how you know any of the above is actually working
```

- **Model** — a very capable, very inconsistent "employee" you can't retrain but can instruct, equip, and manage
- **Context** — the highest-leverage lever in AI engineering; almost everything (RAG, prompting, memory) is context engineering — deciding what enters the model's field of view at generation time
- **System** — orchestration, guardrails, observability, cost/latency economics. The model is not your product; the system around the model is
- **Evaluation** — the meta-principle. Traditional software has tests to catch regressions; AI engineering has to build that safety net deliberately (evaluation-driven development)

---

## 2. AI Engineering vs. Agentic AI vs. Generative AI vs. AI Agents

These terms get used interchangeably in marketing, but they describe different things.

| Term | Simple definition |
|---|---|
| **Generative AI** | AI that creates new stuff — text, images, code, audio. You give it a prompt, it makes something new. It doesn't act or plan, it just generates. *Example: ChatGPT writing a poem.* |
| **AI Agent** | An AI that can actually *do* things, not just talk. It uses tools, takes steps, and checks its own results, continuing until the task is done. *Example: an AI that books your flight by itself.* |
| **Agentic AI** | Not a tool itself — a *style* of behavior. It describes how independent and self-directed a system is. Low agentic = just answers you. High agentic = plans and acts on its own. It's a spectrum, not a fixed thing. |
| **AI Developer / AI Engineer** | The human who builds AI-powered apps — connects models, tools, data, and logic together. They don't train models; they use existing ones smartly. Their job: make AI actually useful and reliable in a product. |

### How to hold these together

- **Generative AI** = the *raw capability* (a model that produces content)
- **AI Agent** = generative AI + tools + a decision loop (a concrete system you build)
- **Agentic AI** = the *adjective* describing how autonomous that system's behavior is — you can say "I built an AI agent that behaves in a highly agentic way"
- **AI Engineer** = the *human role* that builds and ships all of the above

**Quick sanity check:** if someone says "agentic AI," ask *"agentic as opposed to what — a single LLM call, or a fixed pipeline?"* — that's usually what they're pointing at.

---

## 3. Tokens and Tokenizers in Detail

### The core problem a tokenizer solves

Models operate on numbers, not letters. Before any text reaches the neural network, a **tokenizer** converts it into a sequence of integers, and converts the model's output integers back into text.

### Why not split by word, or by character?

- **Character-level tokenization** — tiny vocabulary (~100 symbols), but sequences become enormous ("understanding" → 13 tokens instead of 1–2). Long sequences are expensive and harder to reason over.
- **Word-level tokenization** — short sequences, but an unmanageably huge vocabulary (hundreds of thousands of word forms), and you'd still hit unknown words constantly (typos, slang, other languages).

The industry's middle ground is **subword tokenization**.

### How subword tokenization works (Byte Pair Encoding)

Most modern tokenizers (GPT's tiktoken, Claude's tokenizer, Llama's SentencePiece) use a variant of **Byte Pair Encoding (BPE)**:

1. Start with every individual character as its own token
2. Scan a massive training corpus and find the *most frequently occurring pair* of adjacent tokens
3. Merge that pair into a single new token
4. Repeat this merging process tens of thousands of times

**Result:** common whole words ("the", "and", "running") end up as single tokens because they appeared together often enough to be fully merged. Rare words get left as multiple subword pieces — "unbelievable" might stay split as `un` + `believ` + `able`.

### The tokenization pipeline

```
Raw text            "unbelievable results"
   ↓
Subword tokens       un | believ | able | results
   ↓
Token IDs            4443 | 17527 | 481 | 3059
   ↓
Embedding vectors    (what the model actually processes)
```

### The vocabulary

A trained tokenizer ships with a fixed **vocabulary** — a lookup table mapping every learned subword chunk to an integer ID. Typical vocab sizes run 50,000–200,000 entries, decided once during tokenizer training (before the model itself is trained). The model never sees raw text — only these fixed integer IDs — and it can only ever output IDs from that same fixed set.

### Practical implications

| Behavior | Why it happens |
|---|---|
| **Numbers tokenize awkwardly** | Digits often split unintuitively ("1234" → "123"+"4" or "12"+"34"), part of why LLMs struggle with arithmetic — they pattern-match on token sequences, not true numeric values |
| **Non-English text often costs more tokens** | If the tokenizer's merge rules were trained mostly on English, other languages get chopped into smaller, less-merged pieces — the same sentence in Hindi or Japanese can cost 2–3x more tokens than English |
| **Code has its own quirks** | Whitespace, indentation, and symbols tokenize differently across tokenizers — models with more code in their training mix tend to be cheaper/better at code |
| **Every provider has its own tokenizer** | GPT's `tiktoken`, Claude's tokenizer, Llama's SentencePiece all use different vocabularies — token counts for identical text differ across providers, so you can't reuse one provider's counting library to estimate another's cost |

### Special tokens

Beyond regular text tokens, the vocabulary reserves IDs for structural markers: a "start of sequence" marker, an "end of turn" marker, and role delimiters separating system/user/assistant messages. These are what let a chat model distinguish "the user is talking" from "I am talking" — encoded as special tokens surrounding each message, not inferred from wording.

**Rule of thumb:** 1 token ≈ ¾ of an English word. Everything — cost, context limits, speed — is measured in tokens, not words or characters.

---

## 4. Temperature, Top-p, Context Window, Max Tokens, Hallucination

### Temperature

Controls how random or "creative" the output is — a **risk-taking dial**.

- **Low temperature (0–0.3):** picks the most likely next token almost every time → focused, predictable, repeatable
- **High temperature (0.7–1+):** willing to pick less-likely tokens → more varied, creative, sometimes weirder

**When to use what:** low for factual Q&A, code, data extraction. High for brainstorming, creative writing, varied outputs.

### Top-p (nucleus sampling)

A different lever for controlling randomness: instead of a dial, it restricts the *pool* of tokens the model is even allowed to choose from — "only consider the smallest set of tokens whose combined probability adds up to p%."

Example: top-p = 0.9 means the model only samples from the most likely tokens that together make up 90% of the probability mass, ignoring the unlikely long tail.

> Temperature turns randomness up/down everywhere. Top-p narrows or widens the pool of options. They're often used together, but usually you tune one, not both.

### Context window

The total amount of text (in tokens) the model can "see" at once — the entire conversation, system prompt, documents, everything combined. Think of it as **short-term memory with a hard limit**; once exceeded, older content has to be dropped, summarized, or truncated.

**Why it matters:** this is why long conversations "forget" early details, and why RAG exists — to fetch only the relevant chunk of a large document instead of stuffing everything in. It's also computationally expensive: the attention mechanism compares every token against every other token, so cost grows quadratically with sequence length.

### Max tokens

A hard cap you set on **how long the model's response can be**. If the natural answer would run longer, it gets cut off mid-sentence at that limit — it's a configuration limit, not a model failure.

### Hallucination

When the model states something **confidently, fluently, and wrong** — a fake citation, an invented fact, a function that doesn't exist.

This isn't a "bug" in the traditional sense — the model doesn't look things up, it generates the statistically likely next token. Most of the time that produces true things (because true things are common in training data), but nothing structurally stops it from generating a fluent, plausible-sounding falsehood when it doesn't actually "know." There is no lookup step anywhere in the architecture — a wrong answer is structurally indistinguishable from a right one unless something external checks it.

**Why it matters:** you can't fully trust raw model output for facts — you need grounding (RAG), citations, or verification steps for anything where correctness matters.

### How these connect

| Concept | What it controls |
|---|---|
| Temperature / Top-p | Randomness of the *final token sampling step* only — everything before that (embeddings, attention) is deterministic given the same input |
| Max tokens | Length of the response — a hard cutoff, not intelligent wrap-up |
| Context window | How much the model can see at once — the boundary of its short-term memory |
| Hallucination | A structural failure mode — fluent generation without a fact-checking mechanism |

---

## 5. Transformer Architecture and Attention

The transformer is the architecture behind all modern LLMs. At the core, it's a **next-token prediction machine** — everything impressive it does emerges from repeatedly doing one thing well: given some text, predict what token comes next.

### Step 1 — Embeddings

Each token ID gets looked up in an **embedding table** and converted into a vector — a long list of numbers representing that token's meaning in a geometric space. Tokens with related meanings end up with vectors pointing in similar directions. This is a learned lookup table, tuned during training so meaning becomes geometry.

### Step 2 — Positional information

Embeddings alone don't capture token *order* — "dog bites man" and "man bites dog" would otherwise look identical. The model adds **positional information** to each token's embedding, encoding where in the sequence it sits.

### Step 3 — Self-attention: the core mechanism

For every token, the model asks: *"which other tokens in this sequence should I pay attention to, in order to understand my own meaning right now?"*

Every token produces three vectors:

- **Query** — "what am I looking for?"
- **Key** — "what do I have to offer?"
- **Value** — "what information do I actually contain?"

Each token's Query is compared against every other token's Key (via a dot product), producing a **relevance score** for every pair. Those scores become weights (via softmax, summing to 1), and each token's new representation becomes a weighted blend of every other token's Value, weighted by relevance.

**Example:** In *"The trophy didn't fit in the suitcase because **it** was too big,"* when the model processes "it," its Query vector strongly matches the Key vector for "trophy" rather than "suitcase" — so "it" absorbs mostly "trophy"'s Value information, effectively resolving the reference.

```
Query token "it"  →  compares against every Key in the sentence
                       trophy   : high relevance weight
                       suitcase : low relevance weight
                       (other tokens): very low relevance
```

### Step 4 — Multi-head attention

The model doesn't run this calculation once — it runs many attention operations **in parallel** ("heads"), each with its own learned Query/Key/Value projections. One head might specialize in grammatical relationships, another in resolving pronouns, another in topical relevance. Their outputs are combined afterward, letting one layer capture multiple *kinds* of relationships simultaneously.

### Step 5 — Feed-forward network

After attention mixes information *across* tokens, each token individually passes through a small neural network that processes it *on its own* — no cross-token interaction here. Attention gathers relevant context; the feed-forward step "thinks about" what was gathered. This is also where a large portion of the model's factual/pattern knowledge is believed to live.

### Step 6 — Residual connections and layer normalization

Two "plumbing" details that make deep networks trainable:

- **Residual connections** — each layer's output is *added back* to its input rather than replacing it, so information can persist across many transformations instead of having to survive intact
- **Layer normalization** — rescales values between steps so numbers don't explode or vanish across dozens of stacked layers

### Step 7 — Stacking layers

One full "block" (attention → feed-forward, with residuals and normalization) is one **layer**. Modern LLMs stack dozens of these on top of each other. Roughly speaking, early layers tend to capture more surface-level patterns (grammar, local structure), while later layers capture more abstract relationships (meaning, reasoning-like patterns) — a simplification, not a strict rule.

### Step 8 — Producing the next token

After passing through every layer, the final representation of the last token position is projected into a probability distribution over the entire vocabulary. One token gets sampled (this is where temperature/top-p apply), gets appended to the sequence, and the whole process repeats. This is why generation is inherently sequential — each new token depends on everything generated so far.

### The full pipeline, end to end

```
Text → Tokenizer → Token IDs → Embeddings → + Positional info
   → [Self-Attention → Feed-Forward] × N layers
   → Probability distribution over vocabulary
   → Sample next token (temperature / top-p)
   → Repeat
```

**Key takeaway:** the model has no database of facts it looks up, and no explicit reasoning module. Everything — facts, logic, code ability, conversation — is compressed into billions of numerical weights learned purely from predicting the next token, over and over, on a massive amount of text.

---

## 6. Why LangChain?

### Why it was introduced despite provider SDKs

Provider SDKs (OpenAI's, Anthropic's, etc.) do exactly one thing well: send a request to their model and return a response. Almost nothing else an application needs — chunking/embedding documents, connecting to a vector database, chaining multi-step calls, giving the model tools, managing conversation memory, or swapping providers — is the SDK's job. Every team was solving these same problems from scratch. LangChain's founding idea was to abstract this "plumbing" layer into reusable components.

### Core philosophy: composability

An LLM application is rarely a single model call — it's a **pipeline**: retrieve context → format prompt → call model → parse output → maybe call a tool → maybe loop. LangChain treats each piece of that pipeline as an interchangeable, composable building block — prompt template, retriever, model, output parser, tool — that all share a common interface and can be **chained** together (hence the name).

This is also why LangChain built its own abstraction *on top of* provider APIs rather than wrapping them 1:1: a `ChatModel` behaves identically whether it's calling GPT, Claude, or a local model underneath. Application code doesn't need to know or care which provider sits behind it.

### Why adoption spread so widely

- **Provider independence** — write logic once, swap the underlying model with minimal changes
- **RAG became the dominant pattern** almost overnight — LangChain packaged a multi-step pipeline (load → chunk → embed → store → retrieve → inject) into a few lines
- **Lowered the barrier to entry** — prebuilt patterns (chains, agents, memory) let people build without deeply understanding internals first
- **First-mover network effect** — became the default reference point in tutorials and community content, reinforcing adoption
- **Broad integration ecosystem** — one consistent interface across many vector databases, document loaders, and tools

### The honest trade-off

The same abstraction that drove adoption is also its most common criticism: it can hide what's actually happening in the prompt/API call, making debugging harder and adding unnecessary complexity for simple use cases. Many experienced engineers use LangChain selectively — for genuinely complex orchestration (RAG, agents) — while writing plain API calls for simple tasks, rather than routing everything through it by default.

> **Summary:** LangChain isn't "the way to use LLMs" — it's a toolkit that solves the *orchestration problem* so engineering effort goes into application logic instead of rebuilding chaining, retrieval, and memory plumbing every time.

---

## Quick Reference Table

| Topic | One-line takeaway |
|---|---|
| AI Engineering | Build with existing models, not train new ones |
| Generative AI / AI Agent / Agentic AI / AI Developer | Capability vs. system vs. behavior spectrum vs. human role |
| Token / Tokenizer | Subword chunks + a fixed vocabulary map text to the integers the model actually processes |
| Temperature / Top-p | Control randomness only at the final sampling step |
| Context window | The model's short-term memory limit, in tokens |
| Max tokens | A hard cutoff on response length |
| Hallucination | Fluent but wrong — a structural consequence of generation, not a bug |
| Transformer / Attention | Stacked layers of self-attention + feed-forward, letting every token weigh relevance to every other token |
| LangChain | Solves orchestration (chaining, RAG, memory), not model capability |

---

*Week 1 reference notes — AI Engineering fundamentals.*