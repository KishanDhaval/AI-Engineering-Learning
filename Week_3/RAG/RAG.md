# RAG Pipeline — Reference Notes (Week 3)

RAG = Retrieval-Augmented Generation. Two stages: **Ingestion** (offline, build-time) and **Retrieval** (runtime, per-request), followed by **Generation** (LLM call with injected context).

---

## Full Pipeline Diagram

```
====================================================
 STAGE 1: INGESTION (offline, build-time)
====================================================

  [ Raw Sources ]
  PDF, MD, HTML, CSV, DB rows
          |
          v
  [ Loader ]
  normalize into Document(content, metadata)
          |
          v
  [ Chunker ]
  split into pieces + overlap
          |
          v
  [ Embedding Model ]
  chunk text -> vector
          |
          v
  [ Vector Database ]
  stores {vector, text, metadata}
  builds ANN index (e.g. HNSW)


====================================================
 STAGE 2: RETRIEVAL (runtime, per request)
====================================================

  [ User Query ]
          |
          v
  [ Embed Query ]
  SAME embedding model as ingestion
          |
          v
  [ Similarity Search ]
  top-k search against Vector Database
  (+ optional metadata filter)
          |
          v
  [ Retrieval Strategy ]
  plain top-k / MMR / hybrid / re-rank
          |
          v
  [ Assemble Context ]
  order + dedupe + format with source


====================================================
 STAGE 3: GENERATION
====================================================

  [ Prompt Template ]
  context + original question
          |
          v
  [ LLM Call ]
  Groq / Ollama
          |
          v
  [ Answer ]
  grounded response + citations from metadata
```

---

## Stage 1: Ingestion

### 1.1 Loaders
Normalize raw source formats into a standard `Document{page_content, metadata}` shape. Express analogy: like `express.json()`/`multer` body-parsing middleware — different input formats, one consistent output shape.

**Common loader types:**
- **Text/Markdown loader** — plain `.txt` / `.md` files
- **PDF loader** — handles pages, layout, embedded text (e.g. `PyPDFLoader`)
- **HTML loader** — strips tags, extracts readable content
- **CSV / structured data loader** — row-wise documents, columns become metadata
- **Directory loader** — recursively loads a folder of mixed files
- **Web/URL loader** — fetches + parses live pages
- **Notion / Confluence / Slack loader** — API-based loaders for SaaS knowledge sources
- **Database loader** — pulls rows from SQL/NoSQL as documents

Metadata captured at this stage (source, page number, URL, timestamp) is preserved through the whole pipeline — it's what enables citations and metadata filtering later.

### 1.2 Chunking (Text Splitting)
Needed because embedding models and LLM context windows have token limits, and overly large chunks dilute a vector's semantic precision.

**Chunking strategies:**
- **Fixed-size / character splitting** — cut every N characters; simple, but can cut mid-sentence
- **Recursive character splitting** — tries `\n\n` → `\n` → sentence → word boundaries in order; most common default
- **Token-based splitting** — splits by actual tokenizer output (e.g. tiktoken), not raw characters
- **Semantic chunking** — uses an embedding model to detect meaning shifts and split there; most accurate, most expensive
- **Document-structure-aware splitting** — splits along markdown headers, HTML tags, or code function boundaries

**Key parameters:**
- `chunk_size` — target chunk length (tokens or characters), typically 300–1000 as a starting point
- `chunk_overlap` — overlap between consecutive chunks (e.g. 50–100 tokens) so boundary sentences aren't orphaned

**Trade-off:** too large → blurry/diluted vectors, poor match precision. Too small → loses surrounding context, fragments become meaningless in isolation.

### 1.3 Embeddings
A separate model converts each chunk's text into a fixed-length float vector representing semantic meaning. Same-meaning text → geometrically close vectors, regardless of exact wording.

**Common embedding generation parameters:**
- `model` — which embedding model to use (e.g. `nomic-embed-text`, `text-embedding-3-small`)
- `input` / `prompt` — the text to embed (single string or batch list)
- `input_type` / `task_type` — distinguishes "query" vs "document" embedding mode (some models optimize differently for each)
- `dimensions` — output vector size, if the model supports truncation/config (e.g. 256/512/1536)
- `encoding_format` — e.g. `float` vs base64-encoded
- `truncate` — how to handle input longer than the model's max token limit
- `batch_size` — number of chunks embedded per API call (for throughput on ingestion)
- `normalize` — whether output vectors are L2-normalized (affects which similarity metric is valid)

**Non-negotiable rule:** the embedding model used at ingestion and at query time (retrieval) must always match — different models produce incompatible vector spaces, and mismatches fail silently (no error, just bad results).

### 1.4 Vector Database
Stores `{vector, text, metadata}` and builds an index for fast Approximate Nearest Neighbor (ANN) search — avoids brute-force O(n) comparison across every stored vector.

**Indexing algorithm:** most modern vector DBs use **HNSW** (Hierarchical Navigable Small World) — a multi-layer graph that narrows search from a coarse top layer down to fine-grained lower layers.

**Vector database / library types:**
- **FAISS** — a similarity-search *library*, not a full DB. Extremely fast raw vector math; no built-in persistence, metadata handling, or CRUD — you build that yourself.
- **Chroma** — full embedded/self-hosted vector *database*. Handles storage, metadata, persistence, and querying out of the box.
- **Pinecone** — managed cloud vector DB, serverless, built for production scale
- **Weaviate** — open-source vector DB with hybrid search and schema support built in
- **Qdrant** — open-source, Rust-based, strong filtering + payload support
- **pgvector** — vector similarity as a Postgres extension, useful when data already lives in Postgres
- **Milvus** — distributed vector DB built for large-scale production workloads

**Similarity metrics used for comparison:**
- **Cosine similarity** — angle between vectors, ignores magnitude (most common for text embeddings)
- **Euclidean (L2) distance** — straight-line distance
- **Dot product** — used when vectors are pre-normalized

---

## Stage 2: Retrieval

### 2.1 Query Embedding
User's question is embedded with the exact same embedder/model used during ingestion. This is the single most common source of silent RAG bugs when mismatched.

### 2.2 Similarity Search
Vector DB runs its ANN index against the query vector to return the top-k most similar stored chunks.

**Key parameter — `k`:**
- Too small → misses relevant chunks or answers spanning multiple chunks
- Too large → dilutes LLM context, raises token cost, risks "lost in the middle" (LLM under-attending to buried mid-context info)
- Typical starting point: k = 3–5

### 2.3 Metadata Filtering
Combine similarity ranking with structured filters (e.g. `department`, `year`, `source`) — the vector-search equivalent of a SQL `WHERE` clause layered on top of `ORDER BY` relevance. Used for multi-tenant isolation, access control, and narrowing by date/type/source.

### 2.4 Retrieval Strategies (beyond plain top-k)
Plain "embed → cosine similarity → top-k" is the baseline. Production systems often layer on:

- **Hybrid search** — combines vector similarity with keyword search (BM25/TF-IDF), then merges/re-ranks; catches exact-match terms (IDs, SKUs, codes) that pure semantic search can blur past
- **MMR (Maximal Marginal Relevance)** — balances relevance *and* diversity, avoiding near-duplicate chunks that all cover one sub-topic
- **Re-ranking** — retrieve a larger candidate set (e.g. top-20) cheaply, then re-score with a slower, more accurate cross-encoder model to pick the true top-k
- **Multi-query retrieval** — LLM generates several paraphrased versions of the question, retrieves for each, merges results — reduces sensitivity to exact phrasing
- **Parent-document retrieval** — embed small chunks for precise matching, but return the larger parent section on match, balancing precision with full context

### 2.5 Context Assembly
Before hitting the prompt:
- **Ordering** — most relevant chunks placed at start/end of context (mitigates "lost in the middle")
- **Deduplication** — merge overlapping chunks retrieved due to ingestion-time overlap
- **Formatting** — concatenate with separators, tag each chunk with source metadata for later citation

---

## Stage 3: Generation (Handoff)

Retrieved + formatted context and the original question are injected into a prompt template, then sent to the LLM (Groq/Ollama). LangChain expresses this as an LCEL chain, with the retriever as a `Runnable` step running in parallel with passing the raw question through:

```
rag_chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
```

---

## Key Principles

- **Retrieval quality is usually the bottleneck**, not the LLM — a great LLM fed poor retrieval results underperforms a mediocre LLM fed great retrieval results.
- **Embedder consistency is non-negotiable** — same model for ingestion and query embedding, always.
- **Chunk size/overlap and k are tunable, not fixed** — start with defaults (chunk_size ~500, overlap ~50–100, k ~3–5) and tune against real data.
- **FAISS vs Chroma** — FAISS = low-level vector math library (manual bookkeeping); Chroma = full document store with metadata, persistence, and embedding integration built in.
- **Metadata captured at the loader stage propagates through the entire pipeline** — it's what enables filtering and citations later, so don't discard it early.