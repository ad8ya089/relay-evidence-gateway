"""Provider-neutral contracts; Python acceptance is not wire capability evidence."""

from collections.abc import Awaitable, Callable
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

ErrorKind = Literal[
    "rate_limit_minute",
    "rate_limit_day",
    "rate_limit_unknown",
    "server",
    "timeout",
    "auth",
    "bad_request",
    "disconnect",
    "invalid_response",
    "unsupported_combination",
    "cancelled",
    "callback",
    "persistence",
    "budget_exceeded",
]


class JsonSchemaDefinition(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    name: str
    strict: bool = True
    schema_spec: dict[str, JsonValue] = Field(alias="schema")


class ResponseFormat(BaseModel):
    type: Literal["json_schema", "json_object", "text"]
    json_schema: JsonSchemaDefinition | None = None

    @model_validator(mode="after")
    def schema_required(self) -> "ResponseFormat":
        if (self.type == "json_schema") != (self.json_schema is not None):
            raise ValueError("json_schema definition must accompany exactly the json_schema format")
        return self


class Message(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str


class StreamOptions(BaseModel):
    include_usage: bool = True


class CallMetadata(BaseModel):
    request_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str = "unassigned"
    call_id: str = Field(default_factory=lambda: uuid4().hex)
    kind: str = "draft"
    attempt: int = Field(default=1, ge=1)
    arrival_ns: int | None = Field(default=None, ge=0)


class LLMRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    model: str | None = None
    stream: bool = False
    max_completion_tokens: int = Field(gt=0)
    reasoning_effort: str | None = None
    response_format: ResponseFormat | None = None
    stream_options: StreamOptions | None = None
    metadata: CallMetadata = Field(default_factory=CallMetadata)
    token_reservation: int | None = Field(default=None, gt=0)
    timeout_ms: int | None = Field(default=None, gt=0)
    validation_model: type[BaseModel] | None = Field(default=None, exclude=True)
    schema_stream_policy: Literal["reject", "json_object", "experiment"] = "reject"
    capture_response: bool = False

    @model_validator(mode="after")
    def valid_options(self) -> "LLMRequest":
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options require streaming")
        if (
            self.token_reservation is not None
            and self.token_reservation < self.max_completion_tokens
        ):
            raise ValueError("token reservation must cover at least the completion bound")
        return self

    def to_payload(self, default_model: str) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "messages": [m.model_dump(mode="json") for m in self.messages],
            "model": self.model or default_model,
            "stream": self.stream,
            "max_completion_tokens": self.max_completion_tokens,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.response_format is not None:
            payload["response_format"] = self.response_format.model_dump(
                mode="json", by_alias=True, exclude_none=True
            )
        if self.stream:
            payload["stream_options"] = (self.stream_options or StreamOptions()).model_dump()
        return payload


class Usage(BaseModel):
    tokens_in: int | None = Field(default=None, ge=0)
    tokens_out: int | None = Field(default=None, ge=0)
    tokens_cached: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    queue_time: float | None = Field(default=None, ge=0)
    prompt_time: float | None = Field(default=None, ge=0)
    completion_time: float | None = Field(default=None, ge=0)

    @property
    def counted_tokens(self) -> int | None:
        if self.tokens_in is None or self.tokens_cached is None or self.tokens_out is None:
            return None
        if self.tokens_cached > self.tokens_in:
            return None
        return self.tokens_in - self.tokens_cached + self.tokens_out

    @property
    def complete(self) -> bool:
        return (
            self.counted_tokens is not None
            and self.total_tokens is not None
            and self.tokens_in is not None
            and self.tokens_out is not None
            and self.total_tokens == self.tokens_in + self.tokens_out
        )


class HeaderSnapshot(BaseModel):
    observed_ns: int = Field(ge=0)
    headers: dict[str, str] = Field(default_factory=dict)
    limit_requests_rpd: int | None = None
    remaining_requests_rpd: int | None = None
    limit_tokens_tpm: int | None = None
    remaining_tokens_tpm: int | None = None
    retry_after_ms: float | None = None


class ProviderDiagnostics(BaseModel):
    response_fields: list[str] = Field(default_factory=list)
    usage_locations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    body_trunc: str | None = Field(default=None, max_length=500)
    stream_chunks: list[str] = Field(default_factory=list)


class LLMResult(BaseModel):
    text: str
    model: str
    response_model: str | None = None
    completion_id: str | None = None
    provider_request_id: str | None = None
    status_code: int
    usage: Usage
    usage_complete: bool
    header_snapshot: HeaderSnapshot
    provider_ttft_ms: float | None = None
    first_reasoning_delta_ms: float | None = None
    first_content_delta_ms: float | None = None
    total_ms: float = Field(ge=0)
    structured_output: dict[str, JsonValue] | None = None
    diagnostics: ProviderDiagnostics = Field(default_factory=ProviderDiagnostics)

    @property
    def headers(self) -> dict[str, str]:
        return self.header_snapshot.headers


class ProviderFailure(BaseModel):
    kind: ErrorKind
    status_code: int | None = None
    retry_after_ms: float | None = None
    body_trunc: str = Field(default="", max_length=500)
    headers: dict[str, str] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    total_ms: float = 0
    safe_context: dict[str, JsonValue] = Field(default_factory=dict)


class ProviderError(Exception):
    def __init__(self, info: ProviderFailure) -> None:
        self.info = info
        super().__init__(f"Provider {info.kind} (HTTP {info.status_code})")

    @property
    def kind(self) -> ErrorKind:
        return self.info.kind

    @property
    def status(self) -> int | None:
        return self.info.status_code

    @property
    def retry_after_ms(self) -> float | None:
        return self.info.retry_after_ms

    @property
    def body_trunc(self) -> str:
        return self.info.body_trunc

    @property
    def headers(self) -> dict[str, str]:
        return self.info.headers


class StreamDelta(BaseModel):
    what: Literal["content", "reasoning"]
    text: str
    t_ms: float = Field(ge=0)


DeltaCallback = Callable[[StreamDelta], Awaitable[None]]


class Provider(Protocol):
    async def complete(
        self, request: LLMRequest, *, on_delta: DeltaCallback | None = None
    ) -> LLMResult: ...
