from __future__ import annotations

import json
import os
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(page_title="AI Document Intelligence", page_icon="📄", layout="wide")

st.title("📄 AI Document Intelligence")
st.caption("PDF / image → extraction/OCR → Ollama embeddings → FAISS retrieval → Gemma → validated structured data")

api_url = st.sidebar.text_input("FastAPI URL", os.getenv("API_URL", "http://127.0.0.1:8000"))

uploaded = st.file_uploader(
    "Upload a PDF or image",
    type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tiff"],
)

if uploaded:
    st.write(f"**File:** {uploaded.name}")
    if st.button("Extract structured data", type="primary"):
        with st.spinner("Extracting, embedding, retrieving and reasoning..."):
            try:
                response = requests.post(
                    f"{api_url.rstrip('/')}/extract",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
                    timeout=240,
                )
                response.raise_for_status()
                result = response.json()
            except Exception as exc:
                st.error(f"Request failed: {exc}")
            else:
                data = result["structured_data"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Pages", result["pages"])
                c2.metric("Characters", result["text_length"])
                c3.metric("Chunks", result["chunk_count"])
                c4.metric("Retrieved", len(result["retrieved_chunks"]))

                st.subheader("Structured data")
                st.json(data)

                st.subheader("Retrieved evidence")
                for item in result["retrieved_chunks"]:
                    st.write(f"Chunk {item['index']} · similarity {item['score']:.3f}")

                st.subheader("Extracted text preview")
                st.text_area("Preview", result["extracted_text_preview"], height=250)

                st.download_button(
                    "Download JSON",
                    data=json.dumps(data, indent=2),
                    file_name=f"{Path(uploaded.name).stem}_extracted.json",
                    mime="application/json",
                )
