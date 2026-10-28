import json
from typing import Optional
from pydantic import BaseModel, Field
from integration.strapi_mcp import get_mcp_tools


# ── Minimal Schemas for Read & Update Tools ───────────────────────────
# Replacing Strapi MCP's 14,000-token verbose schemas with lean schemas
# prevents Groq/Gemini TPM rate limits while keeping validation strict.

class EmptyInput(BaseModel):
    """No arguments required."""
    pass


class DocumentIdInput(BaseModel):
    """Fetch or discard a specific record by its stable Strapi documentId."""
    documentId: str = Field(description="The stable documentId of the record in Strapi")


class UpdateJobData(BaseModel):
    """Partial update fields for a job opening. All fields are optional."""
    title: Optional[str] = Field(default=None, description="New job title (min 3 chars)")
    department: Optional[str] = Field(default=None, description="Department name (e.g. Design, Engineering)")
    location: Optional[str] = Field(default=None, description="Location (e.g. Remote, City)")
    employmentType: Optional[str] = Field(
        default=None,
        description="One of: 'Full-time', 'Part-time', 'Contract', 'Internship'",
    )
    experience: Optional[str] = Field(default=None, description="Required experience (e.g. 1-3 years)")
    isOpen: Optional[bool] = Field(default=None, description="true to open position, false to close")
    openings: Optional[int] = Field(default=None, ge=0, description="Total number of openings")


class UpdateJobOpeningInput(BaseModel):
    """Update an existing job opening in Strapi with partial field updates."""
    documentId: str = Field(description="The stable documentId of the job opening to update")
    data: UpdateJobData = Field(description="Fields to update (only include the fields being changed)")


# Output Trimmer: Keep only relevant HR fields
def _trim_output(tool_name: str, raw_output):
    """Strip verbose metadata (HTML, audit timestamps, tracking) to save tokens."""
    try:
        if not (isinstance(raw_output, list) and raw_output and "text" in raw_output[0]):
            return raw_output

        data = json.loads(raw_output[0]["text"])

        # Clean job openings list/get
        if "job-opening" in tool_name:
            items = data.get("results") if "results" in data else ([data] if "documentId" in data else None)
            if items is not None:
                cleaned = [
                    {
                        "documentId": j.get("documentId"),
                        "title": j.get("title"),
                        "department": j.get("department"),
                        "location": j.get("location"),
                        "employmentType": j.get("employmentType"),
                        "experience": j.get("experience"),
                        "openings": j.get("openings"),
                        "isOpen": j.get("isOpen"),
                    }
                    for j in items
                ]
                raw_output[0]["text"] = json.dumps({"jobs": cleaned if "results" in data else cleaned[0]})
                return raw_output

        # Clean job applications list/get
        elif "job-application" in tool_name:
            items = data.get("results") if "results" in data else ([data] if "documentId" in data else None)
            if items is not None:
                cleaned = [
                    {
                        "documentId": a.get("documentId"),
                        "fullName": a.get("fullName"),
                        "email": a.get("email"),
                        "mobileNo": a.get("mobileNo"),
                        "experience": a.get("experience"),
                        "applicationStatus": a.get("applicationStatus"),
                        "position": a.get("position"),
                    }
                    for a in items
                ]
                raw_output[0]["text"] = json.dumps({"applications": cleaned if "results" in data else cleaned[0]})
                return raw_output

    except Exception:
        pass

    return raw_output


def _wrap_tool(tool):
    """Wrap tool coroutine to trim verbose Strapi output."""
    orig_coro = tool.coroutine

    async def wrapped(runtime=None, **kwargs):
        content_blocks, artifact = await orig_coro(runtime=runtime, **kwargs)
        return _trim_output(tool.name, content_blocks), artifact

    tool.coroutine = wrapped
    return tool


# trim strapi args and update with custom schema
async def get_strapi_tools():
    """Retrieve Strapi MCP tools with token-optimized schemas and output trimming."""
    tools = await get_mcp_tools()

    for tool in tools:
        # Read-only list tools take no mandatory arguments
        if tool.name in {"list_job-opening", "list_job-application", "log"}:
            tool.args_schema = EmptyInput

        # Single record lookups only need the documentId
        elif tool.name in {"get_job-opening", "get_job-application", "discard_job-opening_draft"}:
            tool.args_schema = DocumentIdInput

        # Partial updates for job openings (make title, isOpen, openings optional)
        elif tool.name == "update_job-opening":
            tool.args_schema = UpdateJobOpeningInput

        # Wrap read tools to trim huge JSON payloads
        if any(prefix in tool.name for prefix in ("list_", "get_")):
            _wrap_tool(tool)

    return tools