"""Incremental SSE frames with receive timestamps for the answer text bytes."""

import json
import re

from pydantic import BaseModel


class ProtocolError(ValueError):
    pass


class Frame(BaseModel):
    event: str
    data: dict[str, object]
    answer_byte_ns: int | None = None


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("duplicate_sse_json_key")
        result[key] = value
    return result


class SSEDecoder:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.buffer = bytearray()
        self.base = 0
        self.received = 0
        self.stamps: list[tuple[int, int]] = []

    def _stamp(self, offset: int) -> int:
        return next(ns for end, ns in self.stamps if end > offset)

    def _decode(self, raw: bytes) -> Frame | None:
        event = "message"
        fields: list[tuple[bytes, int]] = []
        position = 0
        for line in raw.splitlines(keepends=True):
            content = line.rstrip(b"\r\n")
            if content.startswith(b"event:"):
                event = content[6:].strip().decode("utf-8")
            elif content.startswith(b"data:"):
                start = 6 if content[5:6] == b" " else 5
                fields.append((content[start:], self.base + position + start))
            position += len(line)
        if not fields:
            return None
        body = b"\n".join(value for value, _ in fields)
        try:
            data = json.loads(body, object_pairs_hook=unique_object)
        except (ValueError, UnicodeError):
            raise ProtocolError("malformed_sse_json") from None
        if not isinstance(data, dict):
            raise ProtocolError("non_object_sse_data")
        stamp = None
        if event == "answer" and isinstance(data.get("text"), str) and data["text"]:
            match = re.search(rb'"text"\s*:\s*"', body)
            if match is None:
                raise ProtocolError("answer_text_location_unknown")
            wanted = match.end()
            cursor = 0
            for value, origin in fields:
                if cursor <= wanted < cursor + len(value):
                    stamp = self._stamp(origin + wanted - cursor)
                    break
                cursor += len(value) + 1
        return Frame(event=event, data=data, answer_byte_ns=stamp)

    def feed(self, data: bytes, received_ns: int) -> list[Frame]:
        self.buffer.extend(data)
        self.received += len(data)
        self.stamps.append((self.received, received_ns))
        frames = []
        while match := re.search(rb"\r?\n\r?\n", self.buffer):
            if match.start() > self.maximum:
                raise ProtocolError("sse_frame_too_large")
            frame = self._decode(bytes(self.buffer[: match.start()]))
            if frame:
                frames.append(frame)
            end = match.end()
            del self.buffer[:end]
            self.base += end
            self.stamps = [pair for pair in self.stamps if pair[0] > self.base]
        if len(self.buffer) > self.maximum:
            raise ProtocolError("sse_frame_too_large")
        return frames

    def finish(self) -> None:
        if self.buffer.strip():
            raise ProtocolError("truncated_sse_frame")
