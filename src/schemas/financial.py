"""
Pydantic schema for Financial document extraction.

Covers: income statements, balance sheets, annual reports,
invoices, purchase orders, financial summaries.
"""

from typing import Optional
from pydantic import BaseModel, Field
from .base import ExtractionResultBase


class LineItem(BaseModel):
    """A single financial line item (revenue stream, expense category, etc.)."""

    name: str = Field(description="Label or description of the line item.")
    amount: Optional[float] = Field(
        default=None,
        description="Numeric value. Omit currency symbols; use the document's native unit.",
    )
    period: Optional[str] = Field(
        default=None,
        description="Time period this figure covers, e.g. 'Q1 2024', 'FY2023'.",
    )


class FinancialExtractionResult(ExtractionResultBase):
    """Structured extraction output for financial documents."""

    document_type: str = "financial"

    company_name: Optional[str] = Field(
        default=None,
        description="Legal name of the company or organisation.",
    )
    reporting_period: Optional[str] = Field(
        default=None,
        description="The period covered, e.g. 'Year ended December 31, 2023'.",
    )
    currency: Optional[str] = Field(
        default=None,
        description="ISO 4217 currency code or symbol found in the document, e.g. 'USD', 'CAD', '€'.",
    )
    revenue: Optional[float] = Field(
        default=None,
        description="Total revenue / net sales figure (numeric only).",
    )
    total_expenses: Optional[float] = Field(
        default=None,
        description="Total operating or cost expenses (numeric only).",
    )
    net_income: Optional[float] = Field(
        default=None,
        description="Net income / net profit or loss (numeric only). Use negative for a loss.",
    )
    total_assets: Optional[float] = Field(
        default=None,
        description="Total assets from balance sheet (numeric only), if present.",
    )
    key_line_items: list[LineItem] = Field(
        default_factory=list,
        description="Up to 10 notable line items (major revenue streams, expense categories, etc.).",
    )
