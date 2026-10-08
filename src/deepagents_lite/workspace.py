"""Example capabilities: a per-thread virtual filesystem and a persistent plan.

The tools describe schemas; the wrapper implements them against request.state.
There is no host filesystem backend or special scheduling machinery.
"""

import json
from pathlib import PurePosixPath

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from .middleware import ToolResult


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file from this conversation's virtual workspace."""
    raise RuntimeError("Install the workspace wrapper")


@tool
def write_file(path: str, content: str) -> str:
    """Create or replace a text file in this conversation's virtual workspace."""
    raise RuntimeError("Install the workspace wrapper")


@tool
def list_files() -> str:
    """List the files in this conversation's virtual workspace."""
    raise RuntimeError("Install the workspace wrapper")


@tool
def write_todos(items: list[str]) -> str:
    """Replace the working plan with a list of task descriptions."""
    raise RuntimeError("Install the workspace wrapper")


tools = [read_file, write_file, list_files, write_todos]


def workspace(request, handler):
    if not any(request.tool is t for t in tools):
        return handler(request)
    args = request.tool.args_schema.model_validate(request.tool_call["args"]).model_dump()
    files = dict(request.state.get("files", {}))
    updates = {}
    name = request.tool_call["name"]
    if "path" in args:
        path = PurePosixPath(args["path"])
        if path.is_absolute() or ".." in path.parts or str(path) == ".":
            raise ValueError("Workspace paths must be relative file paths without '..'")
        args["path"] = str(path)
    if name == "write_file":
        files[args["path"]] = args["content"]
        updates, content = {"files": files}, "Written"
    elif name == "read_file":
        content = files[args["path"]]
    elif name == "list_files":
        content = json.dumps(sorted(files))
    else:
        updates, content = {"todos": args["items"]}, "Plan updated"
    return ToolResult(
        ToolMessage(content=content, name=name, tool_call_id=request.tool_call["id"]), updates
    )
