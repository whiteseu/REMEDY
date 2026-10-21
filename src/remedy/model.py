"""Explicitly configured OpenAI-compatible transport; no patient prompt logging."""

from dataclasses import dataclass
import base64
import ipaddress
import json
import mimetypes
import os
from pathlib import Path
import time
from typing import Protocol
import urllib.error
import urllib.parse
import urllib.request


class ModelError(RuntimeError):
    """Transport/provider failure, with no request or response body in the message."""


class Backend(Protocol):
    def complete(self, stage: str, system: str, payload: dict, image: Path | None = None) -> dict | str: ...


@dataclass
class HTTPBackend:
    base_url: str
    model: str
    api_key: str = ""
    timeout: float = 90
    max_tokens: int = 2048
    transport_retries: int = 1

    def __post_init__(self):
        url = urllib.parse.urlsplit(self.base_url)
        if url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError("Set an HTTP(S) API base URL without embedded credentials/query")
        if not self.model.strip() or self.timeout <= 0 or self.max_tokens < 1 or self.transport_retries < 0:
            raise ValueError("Invalid model transport settings")
        self.base_url = self.base_url.rstrip("/")
        self.usage = []
        host = (url.hostname or "").rstrip(".")
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host == "localhost"
        # Windows system proxies may otherwise route local model calls through a relay.
        # Loopback is machine-local; remote endpoints retain normal proxy configuration.
        self._opener = (urllib.request.build_opener(urllib.request.ProxyHandler({}))
                        if loopback else urllib.request.build_opener())

    @classmethod
    def from_env(cls):
        url, model = os.environ.get("REMEDY_BASE_URL"), os.environ.get("REMEDY_MODEL")
        if not url or not model:
            raise ValueError("Set REMEDY_BASE_URL and REMEDY_MODEL explicitly")
        return cls(url, model, os.environ.get("REMEDY_API_KEY", ""))

    def complete(self, stage, system, payload, image=None):
        content = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]
        if image is not None:
            mime = mimetypes.guess_type(str(image))[0]
            if mime not in {"image/png", "image/jpeg", "image/webp"}:
                raise ValueError("Use a PNG, JPEG or WebP query view")
            encoded = base64.b64encode(Path(image).read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        body = json.dumps({"model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": content}]}, allow_nan=False).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        for attempt in range(self.transport_retries + 1):
            req = urllib.request.Request(self.base_url + "/chat/completions", data=body, headers=headers)
            try:
                with self._opener.open(req, timeout=self.timeout) as response:
                    result = json.loads(response.read())
                text = result["choices"][0]["message"]["content"]
                if not isinstance(text, str):
                    raise ModelError("Provider returned non-text completion")
                usage = result.get("usage") or {}
                self.usage.append({"stage": stage,
                                   "prompt_tokens": usage.get("prompt_tokens", 0),
                                   "completion_tokens": usage.get("completion_tokens", 0)})
                return text
            except urllib.error.HTTPError as error:
                retryable = error.code in {429, 500, 502, 503, 504}
                error.close()
                if not retryable or attempt == self.transport_retries:
                    raise ModelError(f"Provider HTTP {error.code}") from None
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == self.transport_retries:
                    raise ModelError("Model endpoint unavailable") from None
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                raise ModelError("Invalid provider response envelope") from None
            time.sleep(min(2 ** attempt, 4))
        raise ModelError("Model request failed")


def parse_object(response):
    if isinstance(response, dict):
        return response
    if not isinstance(response, str):
        raise ValueError("Completion must contain a JSON object")
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("Malformed JSON fence")
        text = "\n".join(lines[1:-1])
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=unique_pairs)
    if not isinstance(value, dict):
        raise ValueError("Completion must contain a JSON object")
    return value
