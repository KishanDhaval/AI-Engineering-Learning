# 💼 Innvonix Careers Hub — AI Recruitment Copilot

An enterprise-grade Talent Acquisition Assistant built with **LangGraph StateGraph**, **Strapi MCP**, **PostgreSQL Memory & Checkpointing**, and **Streamlit**.

Designed specifically for HR teams to search openings, review candidates, draft and publish new vacancies with automated AI responsibility generation, and execute hiring actions under strict **Human-In-The-Loop (HITL)** governance.

---

## 🏛️ Architecture Highlights

- **Stateful Multi-Turn Dialog**: Built on LangGraph `StateGraph` with PostgreSQL persistent memory (`AsyncPostgresSaver` and `AsyncPostgresStore`).
- **Human-In-The-Loop (HITL) Safety**:
  - `create_job_node`: Generates role-specific responsibilities, provides an interactive review card, and supports plain English iterative revisions.
  - `approval_node`: Intercepts mutating actions (`update_*`, `delete_*`, `discard_*`) to ensure human confirmation before changes reach Strapi.
- **Session & Refresh Recovery**: Active threads sync to URL query params (`/?thread=<id>`). If a user refreshes (F5) or returns to a paused session, the exact review card and full chat history are restored seamlessly from PostgreSQL checkpoints.
- **Natural Language Recruiter Memory**: Long-term HR preferences are stored in natural language and compacted using LLM summarization to prevent contradictions and duplicate rules.
- **Lean Token Footprint**: In-process tool schemas and trimmed outputs eliminate verbose HTML and boilerplate, keeping requests well under Groq TPM limits.

---

## 🚀 Quick Start

### 1. Prerequisites & Environment
Ensure your `.env` contains:
```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
STRAPI_API_URL=http://<strapi-host>:1337/mcp
STRAPI_MCP_TOKEN=your_strapi_mcp_bearer_token
DATABASE_URL=postgresql://<user>:<password>@localhost:5432/strapi_hr_assistant
```

### 2. Start Backend & Database
- Ensure PostgreSQL is running and database `strapi_hr_assistant` exists.
- Ensure Strapi backend is running with MCP enabled (`/mcp` endpoint).

### 3. Launch Streamlit Application
```bash
cd strapi-hr-assistant
streamlit run src/app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🎬 End-to-End Demo Script (Continuous 1-Conversation Flow)

Use this exact sequence of prompts in a single chat conversation to showcase all capabilities to stakeholders or clients.

| Step | Action / Recruiter Prompt | What It Demonstrates | Expected System Response |
| :---: | :--- | :--- | :--- |
| **1** | *"Hi! Who are you and how can you help me today?"* | **Brand Persona & Alignment** | Greets warmly as the **Innvonix Careers Hub Copilot**. Explains assistance with vacancies, candidates, and hiring workflows without developer jargon. |
| **2** | *"What job vacancies are currently open at Innvonix?"* | **Read Tool Execution** | Calls `list_job-openings`, retrieves active postings from Strapi, and lists titles, departments, locations, and openings cleanly. |
| **3** | *"Please remember that for all engineering hiring, our default office is Ahmedabad and we prioritize candidates with Python and TypeScript expertise."* | **Recruiter Preference Memory** | Calls `remember_hr_preferences`. Persists the preference in PostgreSQL long-term store. |
| **4** | *"What are my currently saved hiring preferences?"* | **Explicit Memory Recall** | Calls `get_user_preferences` and presents saved preferences in clear bullet points. |
| **5** | *"Let's open a new role for a Backend AI Engineer in Engineering, Ahmedabad with 2 openings and Full-time employment."* | **AI Role Generation & HITL Review** | Triggers `create_job_node`. Generates 4–5 realistic responsibilities and displays the **Job Vacancy Posting — Team Review Required** card. |
| **6** | *(In the Review Card)* Click **✏️ Request Changes in Plain English**, enter: <br>`"Change openings to 3 and add LangGraph & MCP experience to the responsibilities"` and click **Update Draft**. | **Natural Language Revision (No JSON)** | Agent updates the draft fields and rewrites responsibilities in-place, then re-presents the updated card. |
| **7** | *(Mid-Review Test)* **Press F5 (Browser Refresh)** or reload the tab. | **Fault-Tolerant Session Recovery** | Page reloads from `?thread=<id>`. Conversation history and the **Job Review Card reappear instantly** without losing work. |
| **8** | *(In the Review Card)* Click **✅ Approve & Publish Posting**. | **Publishing & Confirmation** | Tool `create_job-opening` executes in Strapi. Agent confirms publication with title, department, location, and openings. |
| **9** | *"Show me the candidate applications we have received."* | **Candidate Pipeline Inspection** | Calls `list_job-applications` and summarizes candidate names, applied positions, and statuses. |
| **10** | *"Update candidate application [Candidate Document ID or Name] to Shortlisted."* | **Mutating Action Interception** | Routes to `approval_node`. Displays an **Action Confirmation** card showing the Document ID and proposed status update. |
| **11** | *(In the Approval Card)* Click **✅ Confirm & Apply**. | **Execution of Mutating Action** | Executes `update_job-application` in Strapi. Confirms candidate is now marked as **Shortlisted**. |

---

## 🧪 Edge Cases & Stress Testing Suite

Test these specific scenarios to verify guardrails, validation, error recovery, and system robustness.

### 1. Incomplete Job Creation (Missing Fields Guardrail)
* **Prompt**:  
  > *"Please create a job opening for Lead Architect."*
* **What Happens**: `create_job_node` detects missing required parameters (`department`, `location`).
* **Expected Result**: Instead of sending a failing request to Strapi or crashing, the assistant politely prompts the recruiter:  
  * *"To create the job opening for Lead Architect, please provide: Department and Location (as well as preferred employment type and number of openings)."*

---

### 2. Contradictory Recruiter Preferences (LLM Compaction)
* **Prompt 1**:  
  > *"Remember that our default hiring location for engineering is Ahmedabad."*
* **Prompt 2**:  
  > *"Actually, all engineering roles are now strictly 100% Remote, not Ahmedabad."*
* **Expected Result**:  
  - The compaction service (`compact_and_save_preferences`) compares the existing preference string with the new instruction.
  - It resolves the conflict by replacing "Ahmedabad" with "100% Remote".
  - Verify with *"What are my saved hiring preferences?"* — only the active Remote rule appears (no contradictions or duplicate bullet points).

---

### 3. Invalid Candidate Application Status (Status Guardrail)
* **Prompt**:  
  > *"Update candidate [Document ID]'s application status to 'AcceptedImmediately'."*
* **What Happens**: In `approval_node`, the status is checked against `VALID_APPLICATION_STATUSES`.
* **Expected Result**:  
  - The invalid status is rejected **before** asking for approval or calling Strapi.
  - The assistant informs the recruiter:  
    * *"Invalid status 'AcceptedImmediately'. Valid statuses are: Applied, Interviewing, Offered, Rejected, Reviewing, Shortlisted, Hired."*

---

### 4. Canceling / Rejecting a Job Posting Draft
* **Prompt**:  
  > *"Draft a job for Office Receptionist in Administration, Ahmedabad, 1 opening, Full-time."*
* **Action**: When the review card appears, enter `Budget frozen for Q4` in the cancel reason field and click **❌ Cancel Posting**.
* **Expected Result**:  
  - The draft is safely aborted without creating any record in Strapi.
  - The assistant responds: *"Job creation was cancelled by HR admin. Reason: Budget frozen for Q4. Please let me know when you'd like to revisit this."*

---

### 5. Canceling a Pipeline Mutating Action
* **Prompt**:  
  > *"Update job opening [Document ID] to closed (isOpen: false)."*
* **Action**: On the confirmation card, enter `Reviewing applicants first` and click **❌ Cancel Action**.
* **Expected Result**:  
  - The tool execution is cancelled.
  - The assistant confirms: *"Action was cancelled. The job vacancy remains open."*

---

### 6. Browser Navigation & Multi-Thread Switching
* **Action**:  
  1. Open a conversation and type a few prompts.
  2. Click **➕ New Conversation** in the sidebar. Notice a new `thread` UUID in the URL.
  3. Click a past conversation in the sidebar list, or paste a previous URL (`/?thread=<old_id>`).
* **Expected Result**:  
  - The application switches threads cleanly.
  - Full history and any paused approval cards are restored for each specific thread without state cross-contamination.

---

### 7. Token Optimization on Large Payloads
* **Prompt**:  
  > *"List all vacancies and list all candidate applications."*
* **What Happens**: Both read tools are called and their outputs are filtered through `_trim_output()`.
* **Expected Result**:  
  - Raw HTML blobs, Strapi timestamps, and internal metadata are stripped.
  - Only essential HR fields (`title`, `department`, `location`, `name`, `status`) are sent back to the model.
  - Request completes quickly without triggering Groq TPM (Tokens Per Minute) limit errors.

---

## 📋 Tool Reference Summary

| Tool Name | Type | Routing Node | Description |
| :--- | :---: | :---: | :--- |
| `list_job-openings` | Read | `tools` | Fetches all job openings from Strapi. |
| `get_job-opening` | Read | `tools` | Fetches details for a single job opening. |
| `list_job-applications` | Read | `tools` | Fetches all candidate applications. |
| `get_job-application` | Read | `tools` | Fetches details for a specific candidate. |
| `get_user_preferences` | Read | `tools` | Retrieves saved recruiter memory from Postgres store. |
| `remember_hr_preferences` | Read/State | `tools` | Saves and compacts recruiter preferences. |
| `create_job-opening` | Create | `create_job_node` | AI generates responsibilities; pauses for HR review before saving to Strapi. |
| `update_job-opening` | Approval | `approval_node` | Pauses for HR confirmation before updating vacancy details in Strapi. |
| `update_job-application` | Approval | `approval_node` | Pauses for HR confirmation before updating candidate status in Strapi. |

---

## 🛡️ Best Practices for Demos
1. Keep the Strapi web dashboard open on another monitor or tab to show real-time synchronization when a job or candidate status is updated.
2. Demonstrate the **F5 refresh** feature during step 7 of the demo script — it highlights production-grade checkpointing and resilience compared to typical chatbot demos.
3. Show how plain English feedback in step 6 adjusts requirements without requiring any JSON editing.