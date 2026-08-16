from __future__ import annotations

import json
import re
from typing import Any

from app.services.ollama_client import OllamaClient


class LLMClient:
    def __init__(self, ollama: OllamaClient | None = None) -> None:
        self.ollama = ollama or OllamaClient()

    def _parse_json(self, content: str) -> dict[str, Any]:
        """
        Parse JSON from an LLM response.

        Handles:
        - normal JSON
        - markdown fenced JSON
        - surrounding text
        - truncated JSON where the closing braces are missing
        """

        content = content.strip()

        # Remove markdown code fences if present.
        content = re.sub(
            r"^```(?:json)?\s*",
            "",
            content,
            flags=re.IGNORECASE,
        )
        content = re.sub(r"\s*```$", "", content).strip()

        # Try normal JSON first.
        try:
            result = json.loads(content)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

        # Try extracting the outermost JSON object.
        start = content.find("{")

        if start == -1:
            raise ValueError(
                f"Gemma returned invalid JSON: {content[:1000]}"
            )

        candidate = content[start:].strip()

        # Try to recover truncated JSON.
        # Count unmatched { and [ brackets and close them.
        repaired = candidate

        open_curly = candidate.count("{") - candidate.count("}")
        open_square = candidate.count("[") - candidate.count("]")

        if open_square > 0:
            repaired += "]" * open_square

        if open_curly > 0:
            repaired += "}" * open_curly

        try:
            result = json.loads(repaired)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

        raise ValueError(
            f"Gemma returned invalid JSON: {content[:1000]}"
        )

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
        )

        return self._parse_json(content)

    def extract_purchase_order(
        self,
        context: str,
    ) -> dict[str, Any]:

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

- total_amount must be a JSON number or null.
- quantity must be a JSON number or null.
- unit_price must be a JSON number or null.
- amount must be a JSON number or null.

Return exactly:

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

Each line item:

{
  "description": "",
  "quantity": null,
  "unit_price": null,
  "amount": null
}

Return only the JSON object.
""".strip()

        user_prompt = f"""
DOCUMENT CONTEXT START

{context}

DOCUMENT CONTEXT END

Extract the requested information.
""".strip()

        content = self.ollama.generate_json(
            system_prompt,
            user_prompt,
            num_predict=512,
        )

        return self._parse_json(content)
