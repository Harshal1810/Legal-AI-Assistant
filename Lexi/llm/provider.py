from __future__ import annotations

from typing import Literal, Optional

ProviderName = Literal['openai', 'groq']


class LLMFactory:
    @staticmethod
    def chat_model(
        provider: ProviderName,
        model: str,
        api_key: str,
        temperature: float = 0.1,
        max_output_tokens: Optional[int] = None,
    ):
        if provider == 'openai':
            try:
                from langchain_openai import ChatOpenAI  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "OpenAI provider selected but langchain-openai is not installed. "
                    "Install: pip install langchain-openai openai"
                ) from e
            kwargs = {'model': model, 'api_key': api_key, 'temperature': temperature}
            if max_output_tokens is not None:
                kwargs['max_tokens'] = max_output_tokens
            return ChatOpenAI(**kwargs)

        try:
            from langchain_groq import ChatGroq  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Groq provider selected but langchain-groq is not installed. "
                "Install: pip install langchain-groq groq"
            ) from e
        kwargs = {'model': model, 'api_key': api_key, 'temperature': temperature}
        if max_output_tokens is not None:
            kwargs['max_tokens'] = max_output_tokens
        return ChatGroq(**kwargs)
