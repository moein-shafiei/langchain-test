"""
Base Pydantic schemas shared across all document types.

ClassificationResult  – produced by the Router Agent.
ExtractionResultBase  – base class for all type-specific extraction models.
"""

from typing import Literal
from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    """Output schema for the Router Agent (GPT-4o-mini classification step)."""

    document_type: Literal["medical", "financial", "generic"] = Field(
        description="The detected category of the document."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the classification (0.0 – 1.0).",
    )
    reasoning: str = Field(
        description="Brief explanation of why this category was chosen."
    )


class ExtractionResultBase(BaseModel):
    """Fields that every extraction result must carry."""

    document_type: str = Field(
        description="The document category used during extraction."
    )
    extraction_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How complete and reliable this extraction is. "
            "Use ≥0.9 when all key fields are found, 0.5–0.8 when some are missing, "
            "<0.5 when major fields could not be located."
        ),
    )
    extraction_notes: str = Field(
        default="",
        description="Any caveats, ambiguities, or fields that could not be extracted.",
    )
