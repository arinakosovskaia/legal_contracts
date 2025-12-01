import json

from contract_cuad.categories import Category
from contract_cuad.classifier import classify_clause, classify_contract_text


class DummyTokenizer:
    def __call__(self, text, add_special_tokens=False, padding=False, truncation=False):
        token_count = len(text.split())
        return {"input_ids": list(range(token_count))}


class StubLLM:
    def __init__(self, outputs):
        self._outputs = outputs
        self.tokenizer = DummyTokenizer()

    def generate_chat(self, messages):
        return self._outputs.pop(0)


CATEGORIES = [
    Category(index=1, name="Confidentiality", description=""),
    Category(index=2, name="Assignment", description=""),
    Category(index=3, name="Change of Control", description=""),
]


def test_classify_clause_returns_top_three_sorted():
    response = json.dumps(
        {
            "categories": [
                {
                    "category": "Assignment",
                    "category_index": 2,
                    "confidence": 0.5,
                    "reason": "Talks about transfer",
                },
                {
                    "category": "Confidentiality",
                    "category_index": 1,
                    "confidence": 0.9,
                    "reason": "Mentions secrecy",
                },
                {
                    "category": "Change of Control",
                    "category_index": 3,
                    "confidence": 0.3,
                    "reason": "Refers to mergers",
                },
            ]
        }
    )
    llm = StubLLM([response])
    result = classify_clause("Sample clause", CATEGORIES, llm)
    assert 1 <= len(result["categories"]) <= 3
    assert result["categories"][0]["confidence"] >= result["categories"][1]["confidence"]
    assert result["categories"][0]["category"] == "Confidentiality"
    assert result["error"] is None


def test_normalize_category_handles_prefixed_names():
    response = json.dumps(
        {
            "categories": [
                {"category": "Category: Confidentiality", "category_index": 1, "confidence": 0.9, "reason": "..."}
            ]
        }
    )
    llm = StubLLM([response])
    result = classify_clause("Sample clause", CATEGORIES, llm)
    assert result["categories"][0]["category"] == "Confidentiality"
    assert result["error"] is None


def test_normalize_category_limits_top_three_and_sorts():
    response = json.dumps(
        {
            "categories": [
                {"category": "Confidentiality", "category_index": 1, "confidence": 0.4, "reason": ""},
                {"category": "Assignment", "category_index": 2, "confidence": 0.9, "reason": ""},
                {"category": "Change of Control", "category_index": 3, "confidence": 0.2, "reason": ""},
                {"category": "Termination", "category_index": 4, "confidence": 0.8, "reason": ""},
                {"category": "Governing Law", "category_index": 5, "confidence": 0.7, "reason": ""},
            ]
        }
    )
    llm = StubLLM([response])
    extended_categories = CATEGORIES + [
        Category(index=4, name="Termination", description=""),
        Category(index=5, name="Governing Law", description=""),
    ]
    result = classify_clause("Sample clause", extended_categories, llm)
    cats = result["categories"]
    assert len(cats) == 3
    confidences = [entry["confidence"] for entry in cats]
    assert confidences == sorted(confidences, reverse=True)
    top_names = [entry["category"] for entry in cats]
    assert top_names == ["Assignment", "Termination", "Governing Law"]


def test_classify_clause_truncated_json_sets_parsing_error_and_falls_back_to_none():
    truncated_response = '{ "categories": [ { "category": "Confidentiality", "category_index": 1, "confidence": 0.9 '
    llm = StubLLM([truncated_response])
    result = classify_clause("This Agreement is confidential...", CATEGORIES, llm)
    assert result["error"] == "PARSING_ERROR"
    assert len(result["categories"]) == 1
    entry = result["categories"][0]
    assert entry["category"] == "NONE"
    assert entry["category_index"] == 0
    assert entry["confidence"] == 0.0


def test_classify_clause_explicit_none_category_is_respected():
    response = json.dumps(
        {
            "categories": [
                {
                    "category": "NONE",
                    "category_index": 0,
                    "confidence": 0.0,
                    "reason": "Clause clearly does not match any CUAD category.",
                }
            ]
        }
    )
    llm = StubLLM([response])
    result = classify_clause("Random boilerplate text", CATEGORIES, llm)
    assert result["error"] is None
    assert len(result["categories"]) == 1
    entry = result["categories"][0]
    assert entry["category"] == "NONE"
    assert entry["category_index"] == 0
    assert entry["confidence"] == 0.0


def test_normalize_category_fuzzy_match_misspelled_name():
    response = json.dumps(
        {
            "categories": [
                {
                    "category": "Confidetiality",
                    "category_index": 1,
                    "confidence": 0.88,
                    "reason": "Misspelled reference to confidentiality.",
                }
            ]
        }
    )
    llm = StubLLM([response])
    result = classify_clause("This Agreement is confidential...", CATEGORIES, llm)
    assert result["error"] is None
    entry = result["categories"][0]
    assert entry["category"] == "Confidentiality"
    assert entry["category_index"] == 1
    assert entry["confidence"] == 0.88


def test_classify_contract_text_sets_primary_category():
    responses = [
        json.dumps(
            {
                "categories": [
                    {
                        "category": "Confidentiality",
                        "category_index": 1,
                        "confidence": 0.8,
                        "reason": "Secret info",
                    }
                ]
            }
        ),
        json.dumps(
            {
                "categories": [
                    {
                        "category": "Assignment",
                        "category_index": 2,
                        "confidence": 0.7,
                        "reason": "Transfer rights",
                    }
                ]
            }
        ),
    ]
    llm = StubLLM(responses)
    text = "Clause one content.\n\nClause two content."
    results = classify_contract_text(
        text,
        CATEGORIES,
        llm,
        fewshot_examples=None,
        show_progress=False,
    )
    assert results[0]["primary_category"] == "Confidentiality"
    assert results[1]["primary_category_index"] == 2
    assert isinstance(results[0]["categories"], list)
    assert results[0]["error"] is None


def test_none_variants_are_normalized_consistently():
    response = json.dumps(
        {
            "categories": [
                {"category": "NONE", "category_index": 0, "confidence": 0.3, "reason": "plain none"},
                {"category": "Category: None", "category_index": 1, "confidence": 0.8, "reason": "prefixed none"},
                {"category": "Category: NONE", "category_index": 2, "confidence": 0.9, "reason": "upper + prefix"},
            ]
        }
    )
    llm = StubLLM([response])
    result = classify_clause("Random clause", CATEGORIES, llm)
    assert result["error"] is None
    assert len(result["categories"]) == 1
    entry = result["categories"][0]
    assert entry["category"] == "NONE"
    assert entry["category_index"] == 0
    assert entry["confidence"] == 0.0


def test_classify_contract_text_mixed_success_and_parsing_error():
    responses = [
        json.dumps(
            {
                "categories": [
                    {
                        "category": "Confidentiality",
                        "category_index": 1,
                        "confidence": 0.95,
                        "reason": "Clearly a confidentiality clause.",
                    }
                ]
            }
        ),
        '{ "categories": [ { "category": "Assignment", "category_index": 2, "confidence": 0.7 ',
    ]
    llm = StubLLM(responses)
    text = "Clause one content.\n\nClause two content."
    results = classify_contract_text(
        text,
        CATEGORIES,
        llm,
        fewshot_examples=None,
        show_progress=False,
    )
    assert results[0]["error"] is None
    assert results[1]["error"] == "PARSING_ERROR"


def test_classify_clause_retries_with_strict_prompt():
    responses = [
        json.dumps(
            {
                "categories": [
                    {"category": "Trademark Usage", "category_index": 1, "confidence": 0.9, "reason": "hallucinated"}
                ]
            }
        ),
        json.dumps(
            {
                "categories": [
                    {
                        "category": "Confidentiality",
                        "category_index": 1,
                        "confidence": 0.85,
                        "reason": "Valid CUAD label on retry.",
                    }
                ]
            }
        ),
    ]
    llm = StubLLM(responses)
    result = classify_clause("Clause mentioning confidentiality obligations.", CATEGORIES, llm)
    assert result["categories"][0]["category"] == "Confidentiality"
    assert result["error"] is None
