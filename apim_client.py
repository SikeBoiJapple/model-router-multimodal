from __future__ import annotations

import logging
from typing import Any

import httpx

from config_loader import Settings

logger = logging.getLogger(__name__)


class APIMClientError(RuntimeError):
    pass


class APIMClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def generate(
        self,
        provider: str,
        model: str,
        prompt: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        provider_conf = self.settings.providers.get(provider)
        if provider_conf is None:
            raise APIMClientError(f"Provider '{provider}' is not configured")

        payload = self._build_payload(provider, model, prompt, options or {})
        headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": provider_conf.subscription_key,
        }

        params: dict[str, str] = {}
        if provider_conf.api_version:
            params["api-version"] = provider_conf.api_version

        try:
            async with httpx.AsyncClient(timeout=provider_conf.timeout_seconds) as client:
                response = await client.post(
                    provider_conf.endpoint,
                    json=payload,
                    headers=headers,
                    params=params,
                )

            response.raise_for_status()
            data = response.json()
            return {
                "output": self._normalize_text(provider, data),
                "raw": data,
            }
        except httpx.TimeoutException as exc:
            raise APIMClientError(
                f"Timeout calling APIM provider '{provider}'"
            ) from exc
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            raise APIMClientError(
                f"APIM provider '{provider}' returned {exc.response.status_code}: {body}"
            ) from exc
        except (ValueError, KeyError, TypeError) as exc:
            logger.exception("Response normalization failure")
            raise APIMClientError(
                f"Unexpected response format from provider '{provider}'"
            ) from exc
        except httpx.HTTPError as exc:
            raise APIMClientError(f"HTTP error calling provider '{provider}': {exc}") from exc

    @staticmethod
    def _build_payload(
        provider: str,
        model: str,
        prompt: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        if provider == "openai":
            return {
                "model": model,
                "input": prompt,
                **options,
            }

        if provider == "claude":
            base = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }
            base.update(options)
            return base

        if provider == "gemini":
            base = {
                "model": model,
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            }
            base.update(options)
            return base

        raise APIMClientError(f"Unsupported provider '{provider}'")

    @staticmethod
    def _normalize_text(provider: str, payload: dict[str, Any]) -> str:
        if provider == "openai":
            if "output_text" in payload:
                return str(payload["output_text"])
            choices = payload.get("choices", [])
            if choices and isinstance(choices, list):
                first_choice = choices[0]
                message = first_choice.get("message", {})
                text_part = message.get("content")
                if text_part is not None:
                    return str(text_part)
            output = payload.get("output", [])
            if output and isinstance(output, list):
                for item in output:
                    content = item.get("content", [])
                    if content and isinstance(content, list):
                        for part in content:
                            if not isinstance(part, dict):
                                continue
                            text_part = part.get("text")
                            if text_part is not None:
                                return str(text_part)
            logger.error(
                "OpenAI response format not recognized. Top-level keys=%s",
                list(payload.keys()),
            )

        if provider == "claude":
            content = payload.get("content", [])
            if content and isinstance(content, list):
                text_part = content[0].get("text")
                if text_part is not None:
                    return str(text_part)

        if provider == "gemini":
            candidates = payload.get("candidates", [])
            if candidates and isinstance(candidates, list):
                first = candidates[0]
                cand_content = first.get("content", {})
                parts = cand_content.get("parts", [])
                if parts and isinstance(parts, list):
                    text_part = parts[0].get("text")
                    if text_part is not None:
                        return str(text_part)

        raise ValueError("Could not extract text from provider response")
