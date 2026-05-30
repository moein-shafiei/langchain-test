"""
Pydantic schema for Generic document extraction.

Used as a catch-all for documents that do not match the medical
or financial categories (e.g. legal agreements, research reports,
policy documents, correspondence).
"""

from typing import Any, Optional
from pydantic import BaseModel, Field
from .base import ExtractionResultBase


class NamedEntity(BaseModel):
    """A named entity detected in the document."""

    text: str = Field(description="The entity text as it appears in the document.")
    entity_type: Optional[str] = Field(
        default=None,
        description="Type hint: PERSON, ORG, LOCATION, DATE, PRODUCT, etc.",
    )


class GenericExtractionResult(ExtractionResultBase):
    """Structured extraction output for generic / unclassified documents."""

    document_type: str = "generic"

    title: Optional[str] = Field(
        default=None,
        description="Document title or heading.",
    )
    summary: Optional[str] = Field(
        default=None,
        description="Two-to-four sentence summary of the document's purpose and main content.",
    )
    key_entities: list[NamedEntity] = Field(
        default_factory=list,
        description="Important named entities: people, organisations, locations, products.",
    )
    dates_mentioned: list[str] = Field(
        default_factory=list,
        description="All significant dates found in the document, in the format found.",
    )
    extracted_fields: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Flexible key-value map for any domain-specific fields that stand out "
            "(e.g. contract numbers, reference IDs, totals, policy terms). "
            "Keep keys descriptive and lowercase_snake_case."
        ),
    )
