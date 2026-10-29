# graph/nodes.py
# Core nodes for the HR assistant StateGraph:
#   1. model_node      → Calls the LLM with conversation history
#   2. create_job_node → Validates job fields, generates responsibilities, pauses for review
#   3. approval_node   → Pauses for HR approval on update/delete actions

import json
from typing import Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt
from langsmith import traceable

from graph.routing import classify_tool

# Valid enums for Strapi fields
VALID_EMPLOYMENT_TYPES = {"Full-time", "Part-time", "Contract", "Internship"}
VALID_APPLICATION_STATUSES = {"Applied", "Reviewed", "Shortlisted", "Rejected", "Hired"}


# Helper: Normalize job data 
def _extract_job_data(args: dict) -> dict:
    """Unpack job fields whether the model passed them flat or inside 'data'."""
    data = args.get("data", args)

    emp_type = data.get("employmentType") or data.get("employment_type", "Full-time")
    if emp_type not in VALID_EMPLOYMENT_TYPES:
        emp_type = "Full-time"

    try:
        openings = int(data.get("openings", 1))
    except (ValueError, TypeError):
        openings = 1

    return {
        "title": str(data.get("title", "")).strip(),
        "department": str(data.get("department", "")).strip(),
        "location": str(data.get("location", "")).strip(),
        "employmentType": emp_type,
        "experience": str(data.get("experience", "Not specified")).strip(),
        "openings": openings,
        "isOpen": bool(data.get("isOpen", True)),
    }


# Helper: Generate Role Responsibilities 
@traceable(name="generate_job_responsibilities")
async def _generate_responsibilities(model, job_data: dict) -> list[str]:
    """Draft 4-5 high-impact, professional responsibilities across 4 key pillars."""
    prompt = f"""You are a senior recruitment specialist drafting comprehensive, professional job responsibilities for Innvonix.

Role: {job_data.get('title')}
Department: {job_data.get('department')}
Experience Level: {job_data.get('experience')}

Generate exactly 4 to 5 high-impact, professional responsibilities. Each bullet point MUST be detailed, substantive, and action-oriented (25-45 words), covering:
1. Core Deliverables & Execution: Specific visual/technical deliverables, campaigns, and day-to-day asset creation.
2. Systems & Quality Standards: Maintaining brand guidelines, design systems, reusable templates, and visual identity.
3. Cross-Functional Collaboration: Working closely with marketing, product, and content teams to translate business requirements.
4. Optimization & Multi-Platform Delivery: Adapting and optimizing assets for diverse digital formats, screen sizes, and performance.

Format rules:
- Return ONLY 4 to 5 bullet points starting with '- '.
- Do NOT include markdown headers, introductory text, or extra commentary."""
    try:
        res = await model.ainvoke([HumanMessage(content=prompt)])
        lines = [
            line.strip().lstrip("-* ").strip()
            for line in res.content.split("\n")
            if line.strip().lstrip("-* ").strip()
        ]
        return lines[:5] or ["Perform core duties aligned with the role."]
    except Exception:
        return ["Drive key departmental initiatives and deliverables."]


# Helper: Apply Natural Language Edits
@traceable(name="apply_natural_language_edits")
async def _apply_natural_language_edits(
    model, job_data: dict, responsibilities: list[str], feedback: str
) -> tuple[dict, list[str]]:
    """Update job details and responsibilities from natural language feedback."""
    prompt = f"""You are helping an HR assistant update a job draft.
Current details: {json.dumps(job_data)}
Current responsibilities: {json.dumps(responsibilities)}
User change request: "{feedback}"

Respond ONLY with a valid JSON object with two keys:
"job_data": (dictionary with title, department, location, employmentType, experience, openings, isOpen)
"responsibilities": (list of updated responsibility strings)"""
    try:
        res = await model.ainvoke([HumanMessage(content=prompt)])
        text = res.content.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
        data = json.loads(text)
        updated_job = _extract_job_data(data.get("job_data", job_data))
        updated_resp = data.get("responsibilities", responsibilities)
        return updated_job, updated_resp
    except Exception:
        return job_data, responsibilities



def _pre_input_guardrail(messages: list) -> Optional[AIMessage]:
    """Intercept candidate application requests and oversized inputs before LLM call."""
    if not messages:
        return None

    last_msg = messages[-1]
    # Only inspect human user messages
    if not (isinstance(last_msg, HumanMessage) or getattr(last_msg, "type", "") in ("human", "user")):
        return None

    content = str(getattr(last_msg, "content", ""))

    # Reject excessively long spam or injection attempts
    if len(content) > 3000:
        return AIMessage(
            content="Your request exceeds the maximum allowed length (3,000 characters). Please provide a more concise inquiry."
        )

    return None


# Tool Call Pruning & Sliding Window 
def _prune_messages(messages: list, max_messages: int = 30) -> list:
    """Trim old tool outputs to save tokens and apply a safe sliding window.

    - The most recent message (if a ToolMessage) is kept full so the model can answer the current turn.
    - Older ToolMessages have their bulky JSON replaced with lightweight receipts.
    - Prevents orphaned tool calls if the sliding window cuts between an AIMessage and a ToolMessage.
    """
    if not messages:
        return []

    pruned = []
    total = len(messages)
    for i, msg in enumerate(messages):
        # Keep the very last message untouched
        if i == total - 1:
            pruned.append(msg)
            continue

        # Evict bulky database payloads from older ToolMessages
        if isinstance(msg, ToolMessage) or getattr(msg, "type", "") == "tool":
            tool_name = getattr(msg, "name", "tool") or "tool"
            tool_call_id = getattr(msg, "tool_call_id", "")
            pruned.append(
                ToolMessage(
                    content=f"[Historical {tool_name} data was retrieved and processed]",
                    name=tool_name,
                    tool_call_id=tool_call_id,
                )
            )
        else:
            pruned.append(msg)

    # Apply sliding window
    windowed = pruned[-max_messages:]

    # Avoid orphaned tool calls at the start of the window
    while windowed and (isinstance(windowed[0], ToolMessage) or getattr(windowed[0], "type", "") == "tool"):
        windowed.pop(0)

    return windowed


# Node 1: model_node 
async def model_node(state, model, system_prompt: str) -> dict:
    """Run the LLM with conversation history and return its response."""
    # Pillar 2: Pre-input guardrail check
    blocked = _pre_input_guardrail(state["messages"])
    if blocked:
        return {"messages": [blocked]}

    # Pillar 3: Prune old tool outputs & apply sliding window
    prepared_messages = _prune_messages(state["messages"], max_messages=10)
    messages_to_send = [SystemMessage(content=system_prompt)] + prepared_messages

    response = await model.ainvoke(messages_to_send)
    return {"messages": [response]}


# Node 2: create_job_node 
async def create_job_node(state, model) -> dict:
    """Validate job fields, generate responsibilities, and pause for HR review."""
    last = state["messages"][-1]

    create_call = next(
        (c for c in last.tool_calls if classify_tool(c["name"]) == "create"),
        None,
    )
    if not create_call:
        return {"pending_tool_calls": None}

    job_data = _extract_job_data(create_call.get("args", {}))

    # Guardrail: Check required fields
    missing_fields = []
    if not job_data["title"] or len(job_data["title"]) < 3:
        missing_fields.append("title (at least 3 characters)")
    if not job_data["department"]:
        missing_fields.append("department")
    if not job_data["location"]:
        missing_fields.append("location")

    if missing_fields:
        return {
            "messages": [
                ToolMessage(
                    tool_call_id=create_call["id"],
                    name=create_call["name"],
                    content=(
                        f"Cannot create job opening yet. "
                        f"Missing required fields: {', '.join(missing_fields)}. "
                        f"Please ask the user for these details first."
                    ),
                )
            ],
            "pending_tool_calls": None,
        }

    # Step: Generate role-specific responsibilities
    responsibilities = await _generate_responsibilities(model, job_data)

    # Human-In-The-Loop: Interactive review loop
    while True:
        user_response = interrupt({
            "type": "job_draft_review",
            "tool_name": create_call["name"],
            "draft": job_data,
            "responsibilities": responsibilities,
        })

        action = user_response.get("action")
        # Backwards compatibility if user sends boolean approved
        if action is None:
            action = "approve" if user_response.get("approved") else "reject"

        # Case 1: Natural Language edit request
        if action == "edit":
            feedback = user_response.get("feedback", "")
            if feedback:
                job_data, responsibilities = await _apply_natural_language_edits(
                    model, job_data, responsibilities, feedback
                )
            continue  # Re-interrupt with updated draft

        # Case 2: Full approval with responsibilities
        elif action == "approve":
            job_data["responsibilities"] = [{"text": r} for r in responsibilities]
            create_call["args"] = {"data": job_data}
            return {
                "messages": [AIMessage(content="", tool_calls=[create_call])],
                "pending_tool_calls": [create_call],
            }

        # Case 3: Approval without responsibilities
        elif action == "approve_without_responsibilities":
            job_data["responsibilities"] = []
            create_call["args"] = {"data": job_data}
            return {
                "messages": [AIMessage(content="", tool_calls=[create_call])],
                "pending_tool_calls": [create_call],
            }

        # Case 4: Rejection / Cancellation
        else:
            reason = user_response.get("reason", "Cancelled by you")
            return {
                "messages": [
                    ToolMessage(
                        tool_call_id=create_call["id"],
                        name=create_call["name"],
                        content=f"The user cancelled this job creation. Reason: {reason}. Acknowledge directly to the user (e.g. 'Understood, I have cancelled the job posting as you requested.').",
                    )
                ],
                "pending_tool_calls": None,
            }


def _resolve_record_info(doc_id: str, messages: list) -> dict:
    """Find human-readable details (title, department, applicant name) for a documentId from history."""
    if not doc_id:
        return {}
    for m in reversed(messages):
        if getattr(m, "type", "") == "tool" or isinstance(m, ToolMessage):
            content = getattr(m, "content", "")
            raw_text = ""
            if isinstance(content, str):
                raw_text = content
            elif isinstance(content, list) and content and isinstance(content[0], dict) and "text" in content[0]:
                raw_text = content[0]["text"]

            if not raw_text or not raw_text.startswith("{"):
                continue
            try:
                data = json.loads(raw_text)
                records = data.get("jobs", []) if "jobs" in data else data.get("applications", [])
                if isinstance(records, dict):
                    records = [records]
                for r in records:
                    if r.get("documentId") == doc_id:
                        return r
            except Exception:
                continue
    return {}


# Node 3: approval_node 
def approval_node(state) -> dict:
    """Pause for HR approval on mutating actions (update, delete, publish)."""
    last = state["messages"][-1]

    calls_to_approve = [
        c for c in last.tool_calls
        if classify_tool(c["name"]) == "approval"
    ]
    if not calls_to_approve:
        return {"pending_tool_calls": None}

    # Check application status validity if updating candidate
    for call in calls_to_approve:
        if "job-application" in call["name"]:
            data = call.get("args", {}).get("data", {})
            status = data.get("applicationStatus")
            if status and status not in VALID_APPLICATION_STATUSES:
                return {
                    "messages": [
                        ToolMessage(
                            tool_call_id=call["id"],
                            name=call["name"],
                            content=(
                                f"Invalid applicationStatus '{status}'. "
                                f"Must be one of: {sorted(VALID_APPLICATION_STATUSES)}."
                            ),
                        )
                    ],
                    "pending_tool_calls": None,
                }

    # Enrich actions with resolved entity details (title, name, department)
    enriched_actions = []
    for c in calls_to_approve:
        doc_id = c.get("args", {}).get("documentId", "")
        target_info = _resolve_record_info(doc_id, state["messages"])
        enriched_actions.append({
            "tool": c["name"],
            "args": c.get("args", {}),
            "target_info": target_info,
        })

    # Pause for HR approval in UI
    user_response = interrupt({
        "type": "action_approval",
        "actions": enriched_actions,
    })

    approved = user_response.get("approved", False)
    reason = user_response.get("reason", "Cancelled by you")

    if approved:
        return {"pending_tool_calls": calls_to_approve}

    return {
        "messages": [
            ToolMessage(
                tool_call_id=c["id"],
                name=c["name"],
                content=f"The user cancelled this action. Reason: {reason}. Acknowledge directly to the user (e.g. 'Understood, I have cancelled the action as you requested.').",
            )
            for c in calls_to_approve
        ],
        "pending_tool_calls": None,
    }
