import unittest
from main import Retriever, chunk_text, tokenize

class LorevaTests(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("FastAPI + RAG!"), ["fastapi", "rag"])

    def test_chunk_text_empty_and_overlap(self):
        self.assertEqual(chunk_text(""), [])
        text = "word " * 300
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunks))

    def test_retrieval_finds_relevant_passage(self):
        r = Retriever()
        r.add_document("refund", "Refunds are available within thirty days with a receipt.")
        r.add_document("garden", "Tomatoes need sunlight, water, and well-drained soil.")
        results = r.search("refund receipt thirty days")
        self.assertEqual(results[0]["document_id"], "refund")

    def test_replacing_document_does_not_duplicate(self):
        r = Retriever()
        r.add_document("policy", "old policy details")
        r.add_document("policy", "new policy details")
        self.assertEqual(len(r.chunks), 1)
        self.assertIn("new", r.chunks[0]["text"])

    def test_invalid_chunking_and_limit(self):
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=10, overlap=10)
        with self.assertRaises(ValueError):
            Retriever().search("query", 0)

if __name__ == "__main__":
    unittest.main()
