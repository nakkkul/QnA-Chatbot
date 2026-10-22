"""Offline regression tests for QnA Chatbot V2 retrieval."""

import unittest

import faiss
import numpy as np

from retrieval import bm25_scores, retrieve_chunks, search_tokens


class FakeEmbeddingModel:
    """Return a fixed query vector without loading a real model."""

    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0]], dtype=np.float32)


class RetrievalTests(unittest.TestCase):

    def setUp(self):
        self.chunks = [
            {
                "text": (
                    "Copyright and publication information. "
                    "All rights reserved by the university."
                ),
                "page": 2,
                "source": "test-guide.pdf",
            },
            {
                "text": (
                    "Self-learning materials are provided in printed "
                    "and electronic booklets for students."
                ),
                "page": 9,
                "source": "test-guide.pdf",
            },
            {
                "text": (
                    "The programme has examinations in June and December "
                    "for the corresponding semester courses."
                ),
                "page": 7,
                "source": "test-guide.pdf",
            },
            {
                "text": (
                    "Section 1.2 Medium of Instruction: "
                    "The medium of instruction is English. "
                    "Course material is in English."
                ),
                "page": 6,
                "source": "test-guide.pdf",
            },
        ]

    def test_search_tokens_normalize_words_and_remove_stop_words(self):
        tokens = search_tokens(
            "The credits and policies for class examinations"
        )

        self.assertEqual(tokens, ["credit", "policy", "class", "examination"])

    def test_bm25_ranks_exact_evidence_highest(self):
        query = "Section 1.2 medium of instruction English"

        scores = bm25_scores(query, self.chunks)

        self.assertIn(3, scores)
        self.assertEqual(
            max(scores, key=scores.get),
            3,
            "The chunk containing the exact instruction-language statement "
            "should rank highest lexically.",
        )

    def test_hybrid_ranking_promotes_relevant_page(self):
        # Deliberately make semantic similarity rank Page 2 first and Page 6
        # last. The exact keyword match should promote Page 6 in the hybrid
        # ranking.
        index = faiss.IndexFlatIP(2)
        vectors = np.array(
            [
                [1.00, 0.00],  # Page 2: highest semantic rank
                [0.95, 0.05],  # Page 9
                [0.90, 0.10],  # Page 7
                [0.10, 1.00],  # Page 6: lowest semantic rank
            ],
            dtype=np.float32,
        )
        index.add(vectors)

        results = retrieve_chunks(
            "Section 1.2 medium of instruction English",
            index,
            self.chunks,
            FakeEmbeddingModel(),
        )

        self.assertEqual(
            results[0]["page"],
            6,
            "Hybrid retrieval should promote the exact evidence chunk.",
        )
        self.assertIn("hybrid_score", results[0])
        self.assertIn("similarity", results[0])
        self.assertEqual(results[0]["source"], "test-guide.pdf")

    def test_retrieval_returns_at_most_four_chunks(self):
        five_chunks = self.chunks + [
            {
                "text": "Additional information about university services.",
                "page": 20,
                "source": "test-guide.pdf",
            }
        ]

        index = faiss.IndexFlatIP(2)
        vectors = np.array(
            [
                [1.00, 0.00],
                [0.90, 0.10],
                [0.80, 0.20],
                [0.70, 0.30],
                [0.60, 0.40],
            ],
            dtype=np.float32,
        )
        index.add(vectors)

        results = retrieve_chunks(
            "medium of instruction English",
            index,
            five_chunks,
            FakeEmbeddingModel(),
        )

        self.assertEqual(len(results), 4)

    def test_empty_index_returns_no_results(self):
        index = faiss.IndexFlatIP(2)

        results = retrieve_chunks(
            "any question",
            index,
            self.chunks,
            FakeEmbeddingModel(),
        )

        self.assertEqual(results, [])

    def test_empty_chunk_list_returns_no_results(self):
        index = faiss.IndexFlatIP(2)

        results = retrieve_chunks(
            "any question",
            index,
            [],
            FakeEmbeddingModel(),
        )

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)