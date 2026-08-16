from __future__ import annotations

import json
from typing import Any

from app.services.ollama_client import OllamaClient


class LLMClient:
    def __init__(self, ollama: OllamaClient | None = None) -> None:
        self.ollama = ollama or OllamaClient()

    def discover_document_schema(
        self,
        document_text: str,
    ) -> dict[str, Any]:

        system_prompt = """
    You are a document structure detection engine.

    Analyze the supplied document text.

    Identify:
    1. The document type.
    2. The document title.
    3. Important header fields actually present in the document.

    Do not assume the document is an invoice.
    Do not invent fields or values.
    Use only information explicitly present in the text.
    Preserve the meaning of the original document labels.
    Return only valid JSON.
    Do not return Markdown or explanations.

    Return exactly:

    {
    "document_type": "",
    "document_title": "",
    "header_fields": [
        {
        "field_name": "",
        "value": null,
        "confidence": 0.0
        }
    ]
    }
    """.strip()

        user_prompt = f"""
    DOCUMENT TEXT:

    {document_text}

    Identify the document type, title, and important header fields.
    Return JSON only.
    """.strip()

        content = self.ollama.generate_json(
            system_prompt,
            user_prompt,
            num_predict=384,
        ).strip()

        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Schema discovery returned invalid JSON: "
                f"{content[:1000]}"
            ) from exc

        if not isinstance(result, dict):
            raise ValueError(
                "Schema discovery did not return a JSON object."
            )

        return result

    def extract_purchase_order(
        self,
        context: str,
    ) -> dict[str, Any]:
        """
        Extract structured business-document fields from retrieved
        document context.

        The current schema is retained for compatibility with the
        existing Pydantic PurchaseOrderExtraction model.
        """

        system_prompt = """
You are a strict document extraction engine.

Extract ONLY facts explicitly present in the supplied document context.

IMPORTANT RULES:

1. Use only information explicitly present in the context.
2. Never invent or guess values.
3. Missing scalar values must be null.
4. Missing line_items must be [].
5. Return exactly one JSON object.
6. Do not write explanations.
7. Do not write "Okay".
8. Do not use Markdown.
9. Return the complete JSON object.
10. Do not stop before the final closing brace.

TYPE RULES:

- total_amount must be a JSON number or null, never a string.
- quantity must be a JSON number or null.
- unit_price must be a JSON number or null.
- amount must be a JSON number or null.

Return exactly this structure:

{
  "vendor_name": null,
  "buyer_name": null,
  "document_number": null,
  "document_date": null,
  "currency": null,
  "total_amount": null,
  "line_items": [],
  "confidence_notes": []
}

Each line item must use:

{
  "description": "",
  "quantity": null,
  "unit_price": null,
  "amount": null
}

For confidence_notes:

- Mention important fields that were not found.
- Mention ambiguous or conflicting evidence.
- Keep notes short.
""".strip()

        user_prompt = f"""
DOCUMENT CONTEXT START

{context}

DOCUMENT CONTEXT END

Extract the requested information now.
Return only the JSON object.
""".strip()

        content = self.ollama.generate_json(
            system_prompt,
            user_prompt,
            num_predict=256,
        ).strip()

        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Gemma returned invalid JSON: "
                f"{content[:1000]}"
            ) from exc

        if not isinstance(result, dict):
            raise ValueError(
                "Gemma returned JSON, but it was not a JSON object."
            )

        return result