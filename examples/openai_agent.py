"""OPENAI_API_KEY=... uv run --extra openai python examples/openai_agent.py"""

import os
from uuid import uuid4

from langchain_openai import ChatOpenAI

from deepagents_lite import Agent, SQLiteStore
from deepagents_lite.workspace import tools, workspace

agent = Agent(
    ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")),
    tools,
    system_prompt="Make a short plan, do the work, then report the result. Use the workspace.",
    store=SQLiteStore("agent.sqlite"),
    tool_wrappers=[workspace],
)
config = {"configurable": {"thread_id": str(uuid4())}}
state = agent.invoke(
    {"messages": [("user", "Write a short haiku about simple software to poem.txt.")]}, config
)
print(state["messages"][-1].content)
print(state.get("files", {}))
