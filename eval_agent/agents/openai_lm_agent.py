import openai
import logging
import backoff

from .base import LMAgent

logger = logging.getLogger("agent_frame")

# OpenAI SDK v1.0+ compatibility
try:
    # New SDK (>=1.0)
    from openai import APIError, APITimeoutError, RateLimitError, APIConnectionError
    OPENAI_ERRORS = (APIError, APITimeoutError, RateLimitError, APIConnectionError)
    OPENAI_V1 = True
except ImportError:
    # Old SDK (<1.0)
    OPENAI_ERRORS = (
        openai.error.APIError,
        openai.error.Timeout,
        openai.error.RateLimitError,
        openai.error.ServiceUnavailableError,
        openai.error.APIConnectionError,
    )
    OPENAI_V1 = False


class OpenAILMAgent(LMAgent):
    def __init__(self, config):
        super().__init__(config)
        assert "model_name" in config.keys()
        if OPENAI_V1:
            # New SDK uses OpenAI client
            self.client = openai.OpenAI(
                api_key=config.get('api_key'),
                base_url=config.get('api_base'),
            )
        else:
            # Old SDK uses module-level config
            if "api_base" in config:
                openai.api_base = config['api_base']
            if "api_key" in config:
                openai.api_key = config['api_key']

    @backoff.on_exception(
        backoff.fibo,
        OPENAI_ERRORS,
    )
    def __call__(self, messages) -> str:
        if OPENAI_V1:
            # New SDK
            response = self.client.chat.completions.create(
                model=self.config["model_name"],
                messages=messages,
                max_tokens=self.config.get("max_tokens", 512),
                temperature=self.config.get("temperature", 0),
                stop=self.stop_words,
            )
            return response.choices[0].message.content
        else:
            # Old SDK
            response = openai.ChatCompletion.create(
                model=self.config["model_name"],
                messages=messages,
                max_tokens=self.config.get("max_tokens", 512),
                temperature=self.config.get("temperature", 0),
                stop=self.stop_words,
            )
            return response.choices[0].message["content"]