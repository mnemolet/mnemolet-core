from mnemolet_core.ingestion.preprocessor import chunk_text


def test_chunk_text():
    text = " ".join(["word"] * 2700)
    chunks = chunk_text(text, max_length=500)
    assert len(chunks) == 6
    assert all(isinstance(c, str) for c in chunks)
    assert sum(len(c.split()) for c in chunks) == 2700
