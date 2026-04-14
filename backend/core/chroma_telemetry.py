"""
No-op Chroma telemetry implementation.

Used to avoid noisy telemetry errors caused by upstream posthog API changes.
"""

from chromadb.config import System
from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoOpProductTelemetry(ProductTelemetryClient):
    """Drop all telemetry events."""

    def __init__(self, system: System):
        super().__init__(system)

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        return
