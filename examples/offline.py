"""No API key: virtual workspace writes, planning, event streaming, and replay."""

from tempfile import TemporaryDirectory

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from deepagents_lite import Agent, SQLiteStore
from deepagents_lite.workspace import tools, workspace


class DemoModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def main():
    model = DemoModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "write_todos", "args": {"items": ["Write a greeting"]}, "id": "plan"},
                    {
                        "name": "write_file",
                        "args": {"path": "hello.txt", "content": "Hello!"},
                        "id": "file",
                    },
                ],
            ),
            AIMessage(content="Created hello.txt and recorded the plan."),
        ]
    )
    config = {"configurable": {"thread_id": "demo"}}
    with TemporaryDirectory() as directory:
        path = f"{directory}/events.sqlite"
        agent = Agent(model, tools, store=SQLiteStore(path), tool_wrappers=[workspace])
        for event in agent.stream({"messages": [("user", "Write a greeting")]}, config):
            print(f"{event.seq}: {event.kind}")
        # A new store and Agent reconstruct the state solely from disk.
        recovered = Agent(model, store=SQLiteStore(path)).invoke(None, config)
        print(recovered["messages"][-1].content)
        print("Files:", recovered["files"])
        print("Plan:", recovered["todos"])


if __name__ == "__main__":
    main()
