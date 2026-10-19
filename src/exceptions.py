"""Typed errors raised by the gateway. Shared by pricing.py and gateway.py."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from .schemas import Provider


class GatewayError(Exception):
    """Base class for every error raised by the gateway."""


class UnsupportedModelError(GatewayError):
    """The model name is not in the pricing registry. Raised before any provider call."""

    def __init__(self, model: str, supported: Iterable[str]) -> None:
        self.model = model
        self.supported = sorted(supported)
        super().__init__(f"Unsupported model {model!r}. Supported models: {', '.join(self.supported)}")


class GatewayProviderError(GatewayError):
    """A provider call failed. The message is already scrubbed of credentials."""

    def __init__(
        self,
        message: str,
        *,
        provider: Provider,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.status_code = status_code
        self.request_id = request_id
        super().__init__(f"[{provider}] {message}")


class AuthenticationError(GatewayProviderError):
    """The API key is missing, invalid or revoked."""


class RateLimitExceededError(GatewayProviderError):
    def __init__(self, message: str, *, retry_after_s: Optional[float] = None, **kwargs: Any) -> None:
        self.retry_after_s = retry_after_s
        super().__init__(message, **kwargs)


class ProviderConnectionError(GatewayProviderError):
    """The provider could not be reached, or the request timed out."""
