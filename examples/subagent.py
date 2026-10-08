"""Delegation is an ordinary tool calling a child loop, with an isolated transcript."""

import json

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from deepagents_lite import Agent, MemoryStore


@tool
def research(question: str) -> str:
    """Delegate a focused research question to a specialist."""
    raise RuntimeError("Install the delegate wrapper")


class ParentModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def main():
    store = MemoryStore()  # Use SQLiteStore to retain both transcripts across processes.
    child = Agent(
        FakeMessagesListChatModel(
            responses=[AIMessage(content="A loop is enough when execution is sequential.")]
        ),
        store=store,
    )

    def delegate(request, handler):
        if request.tool is not research:
            return handler(request)
        question = research.args_schema.model_validate(request.tool_call["args"]).question
        # Stable for this call, including across a parent retry after child completion.
        key = "child:" + json.dumps(
            [request.runtime.thread_id, len(request.state["messages"]), request.tool_call["id"]]
        )
        config = {**request.runtime.config, "configurable": {"thread_id": key}}
        input = None if store.read(key) else {"messages": [("user", question)]}
        result = child.invoke(input, config, context=request.runtime.context)
        return ToolMessage(
            content=result["messages"][-1].content, tool_call_id=request.tool_call["id"]
        )

    parent = Agent(
        ParentModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "research",
                            "args": {"question": "When is a loop sufficient?"},
                            "id": "r1",
                        }
                    ],
                ),
                AIMessage(content="The specialist recommends a loop for sequential execution."),
            ]
        ),
        [research],
        store=store,
        tool_wrappers=[delegate],
    )
    result = parent.invoke(
        {"messages": [("user", "Research simple agents")]},
        {"configurable": {"thread_id": "parent"}},
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
