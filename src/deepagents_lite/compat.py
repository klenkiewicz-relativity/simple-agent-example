"""Optional adapters for individual synchronous LangChain wrapper methods.

Importing this module is cheap. Calling its factories requires the `compat` extra.
This adapts wrappers, not AgentMiddleware lifecycle/state/graph semantics.
"""

from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

from .middleware import ToolResult


def langchain_model_wrapper(wrapper):
    """Adapt `middleware.wrap_model_call` (including decorated wrapper instances)."""
    from langchain.agents.middleware.types import ModelRequest, ModelResponse

    def adapted(request, handler):
        external = ModelRequest(
            model=request.model,
            messages=request.messages,
            tools=request.tools,
            system_message=request.system_message,
            state=request.state,
            runtime=SimpleNamespace(context=request.runtime.context),
            model_settings=request.model_settings,
        )

        def next_handler(changed):
            if changed.response_format is not None or changed.tool_choice is not None:
                raise NotImplementedError(
                    "response_format and tool_choice need native model wrappers"
                )
            response = handler(
                request.override(
                    model=changed.model,
                    messages=changed.messages,
                    tools=changed.tools,
                    system_message=changed.system_message,
                    model_settings=changed.model_settings,
                )
            )
            return ModelResponse(result=[response])

        result = wrapper(external, next_handler)
        if isinstance(result, AIMessage):
            return result
        if (
            isinstance(result, ModelResponse)
            and result.structured_response is None
            and len(result.result) == 1
            and isinstance(result.result[0], AIMessage)
        ):
            return result.result[0]
        raise NotImplementedError("Only AIMessage or a single-message ModelResponse is supported")

    return adapted


def langchain_tool_wrapper(wrapper):
    """Adapt `middleware.wrap_tool_call`; Command and tool state updates are unsupported."""
    from langgraph.prebuilt.tool_node import ToolCallRequest

    def adapted(request, handler):
        external = ToolCallRequest(
            tool_call=request.tool_call,
            tool=request.tool,
            state=request.state,
            runtime=SimpleNamespace(context=request.runtime.context),
        )

        def next_handler(changed):
            result = handler(request.override(tool_call=changed.tool_call, tool=changed.tool))
            if isinstance(result, ToolResult):
                if result.updates:
                    raise NotImplementedError("Use native wrappers around state-changing tools")
                return result.message
            return result

        result = wrapper(external, next_handler)
        if not isinstance(result, ToolMessage):
            raise NotImplementedError("Only ToolMessage is supported; not LangGraph Command")
        return result

    return adapted
