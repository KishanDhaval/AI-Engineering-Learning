# graph/nodes.py
# Core nodes for the HR assistant StateGraph:
#   1. model_node      → Calls the LLM with conversation history
#   2. create_job_node → Validates job fields, generates responsibilities, pauses for review
#   3. approval_node   → Pauses for HR approval on update/delete actions

import json
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt

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
async def _generate_responsibilities(model, job_data: dict) -> list[str]:
    """Draft 4-5 tailored responsibilities for the role using the LLM."""
    prompt = (
        f"Generate 4 to 5 concise, professional job responsibilities for:\n"
        f"Role: {job_data.get('title')}\n"
        f"Department: {job_data.get('department')}\n"
        f"Experience: {job_data.get('experience')}\n\n"
        f"Return ONLY bullet points, one per line starting with '- '."
    )
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


# Node 1: model_node 
async def model_node(state, model, system_prompt: str) -> dict:
    """Run the LLM with conversation history and return its response."""
    messages_to_send = [SystemMessage(content=system_prompt)] + state["messages"]
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
            reason = user_response.get("reason", "Cancelled by HR admin")
            return {
                "messages": [
                    ToolMessage(
                        tool_call_id=create_call["id"],
                        name=create_call["name"],
                        content=f"Job creation was cancelled by HR admin. Reason: {reason}. Inform user.",
                    )
                ],
                "pending_tool_calls": None,
            }


# Node 3: approval_node 
def approval_node(state) -> dict:
    """Pause for HR approval on mutating actions (update, delete)."""
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

    # Pause for HR approval in UI
    user_response = interrupt({
        "type": "action_approval",
        "actions": [
            {"tool": c["name"], "args": c.get("args", {})}
            for c in calls_to_approve
        ],
    })

    approved = user_response.get("approved", False)
    reason = user_response.get("reason", "Action cancelled by HR admin")

    if approved:
        return {"pending_tool_calls": calls_to_approve}

    return {
        "messages": [
            ToolMessage(
                tool_call_id=c["id"],
                name=c["name"],
                content=f"Action '{c['name']}' was rejected by HR admin. Reason: {reason}.",
            )
            for c in calls_to_approve
        ],
        "pending_tool_calls": None,
    }
