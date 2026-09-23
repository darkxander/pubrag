from pubrag.chunk import approx_tokens, chunk_document, chunk_text, split_units


def test_split_units_separates_paragraphs_then_sentences():
    units = split_units("First one. Second one.\n\nThird paragraph.")
    assert units == ["First one.", "Second one.", "Third paragraph."]


def test_empty_text_yields_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_chunks_respect_target_size():
    text = " ".join(f"Sentence number {i} about genetics." for i in range(200))
    chunks = chunk_text(text, target_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    # Tolérance : une unité est ajoutée avant que la limite soit constatée.
    assert all(approx_tokens(c) <= 100 * 1.6 for c in chunks)


def test_oversized_single_sentence_is_kept_whole():
    """Une phrase plus longue que la cible ne doit pas être perdue ni tronquée."""
    long_sentence = "word " * 500
    chunks = chunk_text(long_sentence, target_tokens=50, overlap_tokens=10)
    assert len(chunks) == 1
    assert chunks[0].strip() == long_sentence.strip()


def test_consecutive_chunks_overlap():
    text = " ".join(f"Sentence {i} is here." for i in range(120))
    chunks = chunk_text(text, target_tokens=60, overlap_tokens=20)
    first_tail = chunks[0].split()[-3:]
    assert " ".join(first_tail) in chunks[1]


def test_chunk_document_prefixes_title_and_sets_ids(documents):
    chunks = chunk_document(documents[0], target_tokens=40, overlap_tokens=5)
    assert chunks
    assert all(c.text.startswith(documents[0].title) for c in chunks)
    assert [c.chunk_id for c in chunks] == [f"25188300:{i}" for i in range(len(chunks))]
