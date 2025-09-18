from __future__ import annotations

import os
import json
from typing import Any, Optional
from pathlib import Path
import re


class AgentSDK:
    """Wrapper for OpenAI Agents SDK with safe fallback to Chat Completions.

    Usage: sdk.run_json(system, user, schema_hint="...", attempts=2)
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        # Load API key from arg, env, or fallback file
        key = api_key or os.getenv("OPENAI_API_KEY")
        if key:
            os.environ["OPENAI_API_KEY"] = key
        self._client = None
        self._agents = None
        self.last_error: Optional[str] = None
        self.last_raw: Optional[str] = None
        self._init()

    def _init(self) -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI()
        except Exception:
            self._client = None
        # Try Agents SDK
        try:
            from openai.agents import Agents  # type: ignore
            if self._client is not None:
                self._agents = Agents(self._client)
        except Exception:
            self._agents = None

    def run_json(
        self,
        system: str,
        user: str,
        *,
        schema_hint: Optional[str] = None,
        attempts: int = 2,
        temperature: float = 0.0,
        timeout_s: float | None = 60.0,
    ) -> dict:
        # Prefer Agents SDK if present, otherwise chat.completions
        # To keep it simple and robust, both paths request JSON-only responses.
        # Agents path (if available)
        if self._agents is not None:
            try:
                # Use agents response with enforced JSON format
                client = self._client
                try:
                    client = self._client.with_options(timeout=timeout_s)  # type: ignore[attr-defined]
                except Exception:
                    pass
                # First attempt: request JSON format
                try:
                    kwargs = {
                        "model": self.model,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    }
                    resp = client.chat.completions.create(**kwargs)
                    content = resp.choices[0].message.content if resp.choices else ""
                    self.last_raw = content or ""
                    return json.loads(content) if content else {}
                except Exception as e1:
                    # Fallback: no response_format; parse flexibly
                    self.last_error = f"agents json_format unsupported, fallback: {e1}"
                    kwargs = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    }
                    resp = client.chat.completions.create(**kwargs)
                    content = resp.choices[0].message.content if resp.choices else ""
                    self.last_raw = content or ""
                    data = _parse_json_flexible(content)
                    if isinstance(data, dict) and data:
                        return data
            except Exception as e:
                self.last_error = f"agents/chat path error: {e}"

        # Chat Completions path
        if self._client is not None:
            last_text = ""
            for i in range(max(1, attempts)):
                try:
                    client = self._client
                    try:
                        client = self._client.with_options(timeout=timeout_s)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    try:
                        kwargs = {
                            "model": self.model,
                            "response_format": {"type": "json_object"},
                            "messages": [
                                {"role": "system", "content": system},
                                {"role": "user", "content": user},
                            ],
                        }
                        resp = client.chat.completions.create(**kwargs)
                        content = resp.choices[0].message.content if resp.choices else ""
                        last_text = content or ""
                        self.last_raw = last_text
                        if last_text.strip():
                            return json.loads(last_text)
                    except Exception as e1:
                        # Fallback without response_format
                        self.last_error = f"chat json_format unsupported, fallback: {e1}"
                        kwargs = {
                            "model": self.model,
                            "messages": [
                                {"role": "system", "content": system},
                                {"role": "user", "content": user},
                            ],
                        }
                        resp = client.chat.completions.create(**kwargs)
                        content = resp.choices[0].message.content if resp.choices else ""
                        last_text = content or ""
                        self.last_raw = last_text
                        data = _parse_json_flexible(last_text)
                        if isinstance(data, dict) and data:
                            return data
                except Exception as e:
                    self.last_error = f"chat path error: {e}"
                user = (user + "\n\nReturn ONLY valid JSON object with no surrounding prose." + (f" Schema: {schema_hint}" if schema_hint else ""))
        return {}

    def diagnostics(self) -> dict:
        return {"last_error": self.last_error, "last_raw": (self.last_raw[:200] + "…") if self.last_raw else None}


def _parse_json_flexible(text: str) -> Any:
    if not text:
        return {}
    t = text.strip()
    # Try fenced code block ```json
    m = re.search(r"```(?:json)?\n(.*?)\n```", t, flags=re.IGNORECASE | re.DOTALL)
    if m:
        body = m.group(1).strip()
        try:
            return json.loads(body)
        except Exception:
            pass
    # Try direct JSON
    if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
        try:
            return json.loads(t)
        except Exception:
            pass
    # Try substring between first { and last }
    start = t.find("{")
    end = t.rfind("}")
    if 0 <= start < end:
        candidate = t[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return {}
