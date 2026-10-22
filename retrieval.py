"""Hybrid retrieval utilities for QnA Chatbot V2."""

import math
import re
from collections import Counter

import numpy as np


TOP_K = 4
RRF_K = 60
LEXICAL_RRF_WEIGHT = 1.3

SEARCH_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "by", "can", "could", "did", "do", "does", "for", "from",
    "had", "has", "have", "how", "i", "in", "is", "it", "its",
    "may", "me", "my", "of", "on", "or", "our", "please",
    "should", "that", "the", "their", "them", "there", "this",
    "those", "to", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "would", "you", "your",
    "according", "guide",
}


def search_tokens(text: str) -> list[str]:
    """Normalize words for lightweight lexical retrieval."""
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    normalized = []

    for token in tokens:
        if len(token) > 5 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif (
            len(token) > 4
            and token.endswith("s")
            and not token.endswith("ss")
        ):
            token = token[:-1]

        if token and token not in SEARCH_STOP_WORDS:
            normalized.append(token)

    return normalized


def bm25_scores(query: str, documents: list[dict]) -> dict[int, float]:
    """Calculate BM25-style keyword relevance for each text chunk."""
    tokenized_docs = [
        search_tokens(document["text"])
        for document in documents
    ]

    document_count = len(tokenized_docs)
    if document_count == 0:
        return {}

    term_counts = [Counter(tokens) for tokens in tokenized_docs]
    document_lengths = [len(tokens) for tokens in tokenized_docs]
    average_length = sum(document_lengths) / document_count

    if average_length == 0:
        return {}

    document_frequency = Counter()
    for tokens in tokenized_docs:
        document_frequency.update(set(tokens))

    query_terms = set(search_tokens(query))
    if not query_terms:
        return {}

    k1 = 1.5
    b = 0.75
    scores = {}

    for document_id, frequencies in enumerate(term_counts):
        length = document_lengths[document_id]
        score = 0.0

        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue

            df = document_frequency[term]
            inverse_document_frequency = math.log(
                1.0 + (document_count - df + 0.5) / (df + 0.5)
            )

            length_normalizer = (
                k1 * (1.0 - b + b * length / average_length)
            )

            score += (
                inverse_document_frequency
                * frequency
                * (k1 + 1.0)
                / (frequency + length_normalizer)
            )

        if score > 0:
            scores[document_id] = score

    return scores


def retrieve_chunks(
    question: str,
    index,
    chunks: list[dict],
    embedding_model,
) -> list[dict]:
    """Fuse semantic similarity and BM25 rankings using RRF."""
    if index is None or not chunks or index.ntotal == 0:
        return []

    query_vector = embedding_model.encode(
        [question],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    query_vector = np.asarray(query_vector, dtype=np.float32)

    # Get a complete semantic ranking so lexical results can move upward.
    semantic_raw_scores, semantic_raw_indices = index.search(
        query_vector,
        index.ntotal,
    )

    semantic_order = [
        int(index_id)
        for index_id in semantic_raw_indices[0]
        if 0 <= index_id < len(chunks)
    ]

    semantic_rank = {
        document_id: rank
        for rank, document_id in enumerate(semantic_order, start=1)
    }

    semantic_scores = {
        int(index_id): float(score)
        for score, index_id in zip(
            semantic_raw_scores[0],
            semantic_raw_indices[0],
        )
        if 0 <= index_id < len(chunks)
    }

    lexical_scores = bm25_scores(question, chunks)
    lexical_order = sorted(
        lexical_scores,
        key=lambda document_id: (
            lexical_scores[document_id],
            semantic_scores.get(document_id, -1.0),
        ),
        reverse=True,
    )

    lexical_rank = {
        document_id: rank
        for rank, document_id in enumerate(lexical_order, start=1)
    }

    hybrid_scores = {}

    for document_id in semantic_order:
        score = 1.0 / (RRF_K + semantic_rank[document_id])

        if document_id in lexical_rank:
            score += (
                LEXICAL_RRF_WEIGHT
                / (RRF_K + lexical_rank[document_id])
            )

        hybrid_scores[document_id] = score

    ranked_ids = sorted(
        hybrid_scores,
        key=lambda document_id: (
            hybrid_scores[document_id],
            semantic_scores.get(document_id, -1.0),
        ),
        reverse=True,
    )

    results = []
    for document_id in ranked_ids[:min(TOP_K, len(chunks))]:
        result = dict(chunks[document_id])
        result["similarity"] = semantic_scores.get(document_id, 0.0)
        result["hybrid_score"] = hybrid_scores[document_id]
        results.append(result)

    return results