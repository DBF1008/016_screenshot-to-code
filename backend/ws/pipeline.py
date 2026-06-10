from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List

from fastapi import WebSocket
from openai.types.chat import ChatCompletionMessageParam

from llm import Llm
from ws.communicator import MessageType, WebSocketCommunicator


@dataclass
class PipelineContext:
    """Context object that carries state through the pipeline"""

    websocket: WebSocket
    ws_comm: WebSocketCommunicator | None = None
    params: Dict[str, Any] = field(default_factory=dict)
    extracted_params: Any = None
    prompt_messages: List[ChatCompletionMessageParam] = field(default_factory=list)
    variant_models: List[Llm] = field(default_factory=list)
    completions: List[str] = field(default_factory=list)
    variant_completions: Dict[int, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def send_message(self):
        assert self.ws_comm is not None
        return self.ws_comm.send_message

    @property
    def throw_error(self):
        assert self.ws_comm is not None
        return self.ws_comm.throw_error


class Middleware(ABC):
    """Base class for all pipeline middleware"""

    @abstractmethod
    async def process(
        self, context: PipelineContext, next_func: Callable[[], Awaitable[None]]
    ) -> None:
        """Process the context and call the next middleware"""
        pass


class Pipeline:
    """Pipeline for processing WebSocket code generation requests"""

    def __init__(self):
        self.middlewares: List[Middleware] = []

    def use(self, middleware: Middleware) -> "Pipeline":
        """Add a middleware to the pipeline"""
        self.middlewares.append(middleware)
        return self

    async def execute(self, websocket: WebSocket) -> None:
        """Execute the pipeline with the given WebSocket"""
        context = PipelineContext(websocket=websocket)

        # Build the middleware chain
        async def start(ctx: PipelineContext):
            pass  # End of pipeline

        chain = start
        for middleware in reversed(self.middlewares):
            chain = self._wrap_middleware(middleware, chain)

        await chain(context)

    def _wrap_middleware(
        self,
        middleware: Middleware,
        next_func: Callable[[PipelineContext], Awaitable[None]],
    ) -> Callable[[PipelineContext], Awaitable[None]]:
        """Wrap a middleware with its next function"""

        async def wrapped(context: PipelineContext) -> None:
            await middleware.process(context, lambda: next_func(context))

        return wrapped
