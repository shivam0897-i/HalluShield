import pytest

pytest.importorskip("fastapi", reason="API extras not installed (pip install -e \".[api]\")")

from fastapi.testclient import TestClient  # noqa: E402

from hallushield.middleware.app import app  # noqa: E402

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_validate_grounded_payload_passes():
    payload = {
        "query": "What is the recommended dose of Metformin?",
        # Fully grounded phrasing (every content word appears in the chunk), so it
        # clears even the strict medical bar under the trivial lexical baseline.
        # A looser paraphrase ("started at" vs "initial dose") correctly lands at
        # WARN here — that gap is what the LettuceDetect grounding signal closes.
        "answer": "The metformin dose is 500mg twice daily with meals.",
        "chunks": [
            {"id": "chunk_1", "text": "Metformin initial dose is 500mg twice daily with meals.",
             "source": "ADA 2024"}
        ],
        "domain": "medical",
    }
    body = client.post("/validate", json=payload).json()
    assert body["verdict"] == "PASS"
    assert body["claims"][0]["supporting_chunk"] == "chunk_1"
    assert "latency_ms" in body


def test_validate_fabricated_payload_not_passed():
    payload = {
        "query": "Metformin dose?",
        "answer": "Metformin should be injected at 5000mg every hour.",
        "chunks": [
            {"id": "chunk_1", "text": "Metformin initial dose is 500mg twice daily."}
        ],
        "domain": "medical",
    }
    body = client.post("/validate", json=payload).json()
    assert body["verdict"] != "PASS"
