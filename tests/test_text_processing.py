from app.services.text_processing import chunk_text, clean_text


def test_clean_text():
    assert clean_text(" a  b\r\n\r\n\r\n c ") == "a b\n\nc"


def test_chunk_text_overlap():
    text = "abcdefghij"
    chunks = chunk_text(text, chunk_size=6, overlap=2)
    assert chunks[0] == "abcdef"
    assert chunks[1].startswith("ef")
