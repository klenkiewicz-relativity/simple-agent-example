import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool

pytest.importorskip("langchain")

from langchain.agents.middleware import wrap_model_call, wrap_tool_call  # noqa: E402

from deepagents_lite import Agent  # noqa: E402
from deepagents_lite.compat import langchain_model_wrapper, langchain_tool_wrapper  # noqa: E402


class Scripted(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_real_langchain_decorators_and_request_overrides():
    seen = []

    @tool
    def double(x: int) -> int:
        """Double a number."""
        return x * 2

    @wrap_model_call
    def prompt(request, handler):
        assert request.runtime.context == "client"
        seen.append("model")
        result = handler(request.override(system_message=SystemMessage(content="Be concise")))
        assert isinstance(result.result[0], AIMessage)
        return result

    @wrap_tool_call
    def change_args(request, handler):
        seen.append("tool")
        return handler(request.override(tool_call={**request.tool_call, "args": {"x": 3}}))

    agent = Agent(
        Scripted(
            responses=[
                AIMessage(content="", tool_calls=[{"name": "double", "args": {"x": 1}, "id": "d"}]),
                AIMessage(content="six"),
            ]
        ),
        [double],
        model_wrappers=[langchain_model_wrapper(prompt.wrap_model_call)],
        tool_wrappers=[langchain_tool_wrapper(change_args.wrap_tool_call)],
    )
    state = agent.invoke(
        {"messages": [("user", "double")]},
        {"configurable": {"thread_id": "compat"}},
        context="client",
    )
    assert state["messages"][2].content == "6"
    assert seen == ["model", "tool", "model"]
