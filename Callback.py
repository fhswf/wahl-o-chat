"""Callback Handler that prints to streamlit."""


from __future__ import annotations

from asyncio import Queue
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler, AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from langchain_community.callbacks.streamlit.mutable_expander import MutableExpander


class GradioCallbackHandler(BaseCallbackHandler):
    """Callback handler that yields events."""

    def __init__(
        self, progress
    ):
        self.queue = Queue()
        self.done = False
        self.progress = progress
        self.count = 0
        self.progress(self.count, "Verarbeite Frage")

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.done:
            raise StopAsyncIteration
        self.count += 1
        self.progress(self.count, "... verarbeite Informationen", total=100)
        return await self.queue.get()
        
    def end_run(self, future):
        print("end_run: ", future)
        self.done = True
        self.queue.put_nowait("end of run")

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print("llm end: ", response)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        pass

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        print("tool start: ", serialized, input_str, kwargs)

    def on_tool_end(
        self,
        output: Any,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        print("tool end: ", output)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        print("tool error: ", error)

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        print("on_text: ", text)

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        print("chain starting: ", str(serialized)[:100], str(inputs)[:100], kwargs)

        self.queue.put_nowait(f"chain start: {inputs}")

    def on_retriever_start(self, serialized: dict[str, Any], query: str, *, run_id: UUID, parent_run_id: UUID | None = None, tags: list[str] | None = None, metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        print("on_retriever_start: ", query, run_id, parent_run_id, tags, metadata)

    def on_retriever_end(self, documents: Sequence[Document], *, run_id: UUID, parent_run_id: UUID | None = None, tags: list[str] | None = None, **kwargs: Any) -> None:
        print("on_retriever_end: ", documents, run_id, parent_run_id, tags)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        print("chain end: ", str(outputs)[:100], kwargs)
        self.queue.put_nowait(f"chain end: {outputs}")

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        print("chain error: ", error)

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        pass

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        pass