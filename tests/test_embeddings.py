from mnemolet_core.embeddings.local_llm_embed import embed_texts


def test_embed_texts_basic():
    texts = ["Hello world", "Another chunk"]
    embeddings = embed_texts(texts)
    assert len(embeddings) == len(texts)
    for xz in embeddings:
        assert isinstance(xz, list)
        assert all(isinstance(x, float) for x in xz)
        assert len(xz) == 384  # dimension of small embedding model


def test_embed_texts_empty():
    embeddings = embed_texts([])
    assert embeddings == []
