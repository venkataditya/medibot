"""Thin wrapper over Groq chat completions with one friendly error type."""

from typing import Any

import groq


class LLMError(RuntimeError):
    """Raised for any failure talking to the LLM; the message is safe to show a user."""


class GroqLLM:
    def __init__(self, model: str, api_key: str | None = None, client: Any | None = None) -> None:
        if client is None:
            if not api_key:
                raise LLMError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
            client = groq.Groq(api_key=api_key)
        self._client = client
        self.model = model

    def complete(self, system: str, user: str, temperature: float = 0, json_mode: bool = False) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(**kwargs)
        except groq.AuthenticationError as e:
            raise LLMError("Groq rejected the API key. Check GROQ_API_KEY in .env.") from e
        except groq.NotFoundError as e:
            raise LLMError(f"Groq model '{self.model}' is not available. Set GROQ_MODEL to a current model.") from e
        except groq.RateLimitError as e:
            raise LLMError("Groq rate limit reached. Wait a moment and try again.") from e
        except groq.APIConnectionError as e:
            raise LLMError("Could not reach Groq. Check your network connection.") from e
        except groq.APIStatusError as e:
            raise LLMError(f"Groq returned an error (HTTP {e.status_code}). Try again shortly.") from e

        content = response.choices[0].message.content
        if not content or not content.strip():
            raise LLMError("The model returned an empty answer. Try rephrasing the question.")
        return content.strip()
