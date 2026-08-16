from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class LineItem(BaseModel):
    description: str = ""
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None


class PurchaseOrderExtraction(BaseModel):
    vendor_name: Optional[str] = None
    buyer_name: Optional[str] = None
    document_number: Optional[str] = None
    document_date: Optional[str] = None
    currency: Optional[str] = None
    total_amount: Optional[float] = None
    line_items: List[LineItem] = Field(default_factory=list)
    confidence_notes: List[str] = Field(default_factory=list)

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip().upper() or None


class RetrievedChunkInfo(BaseModel):
    index: int
    score: float


class ExtractionResponse(BaseModel):
    filename: str
    document_type: str
    pages: int
    text_length: int
    chunk_count: int
    retrieved_chunks: List[RetrievedChunkInfo]
    extracted_text_preview: str
    structured_data: PurchaseOrderExtraction
