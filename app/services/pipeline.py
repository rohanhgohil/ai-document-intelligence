from __future__ import annotations

from app.schemas import PurchaseOrderExtraction
from app.services.llm_client import LLMClient
from app.services.ollama_client import OllamaClient
from app.services.rag_retriever import LocalRAGRetriever
from app.services.text_extractor import extract_text
from app.services.text_processing import chunk_text, clean_text


class DocumentIntelligencePipeline:
    def __init__(self) -> None:
        ollama = OllamaClient()
        self.llm = LLMClient(ollama)
        self.retriever = LocalRAGRetriever(ollama)

    def run(self, file_path: str, filename: str, top_k: int = 1):
        raw_text, pages, document_type = extract_text(file_path)
        cleaned = clean_text(raw_text)
        if not cleaned:
            raise ValueError(
                "No text was extracted from the document."
            )

        # Discover the document type and generic header structure
        header_text = cleaned[:6000]

        schema = self.llm.discover_document_schema(header_text)
        chunks = chunk_text(
            cleaned,
            chunk_size=2500,
            overlap=250,
        )

        if not cleaned:
            raise ValueError(
                "No text was extracted from the document. For scanned PDFs, use an image/OCR input or add PDF OCR as the next enhancement."
            )
        MAX_CHUNKS = 20

        if len(chunks) > MAX_CHUNKS:
            print(
                f"[RAG] Document produced {len(chunks)} chunks. "
                f"Limiting to first {MAX_CHUNKS} for MVP."
            )
            chunks = chunks[:MAX_CHUNKS]
        self.retriever.build(chunks)
        query = (
            "Find the vendor name, buyer name, invoice or document number, "
            "document date, currency, subtotal, tax, total amount, and all line items."
        )
        retrieved = self.retriever.search(query, top_k=2)
        MAX_CONTEXT_CHARS = 2000

        context_parts = []
        total_chars = 0

        for item in retrieved:
            part = f"[similarity={item.score:.3f}]\n{item.text}"

            if total_chars + len(part) > MAX_CONTEXT_CHARS:
                remaining = MAX_CONTEXT_CHARS - total_chars

                if remaining > 200:
                    context_parts.append(part[:remaining])

                break

            context_parts.append(part)
            total_chars += len(part)

        context = "\n\n--- RETRIEVED CHUNK ---\n\n".join(context_parts)

        data = self.llm.extract_purchase_order(context)
        structured = PurchaseOrderExtraction.model_validate(data)

        return {
            "filename": filename,
            "document_type": document_type,
            "pages": pages,
            "text_length": len(cleaned),
            "chunk_count": len(chunks),
            "discovered_schema": schema,
            "retrieved_chunks": [
                {"index": item.index, "score": round(item.score, 4)} for item in retrieved
            ],
            "extracted_text_preview": cleaned[:2000],
            "structured_data": structured.model_dump(),
        }
