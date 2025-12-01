from contract_cuad.classifier import parse_llm_json


def test_parse_clean_json():
    payload = '{"categories": [{"category": "Confidentiality", "category_index": 1, "confidence": 0.9, "reason": "Contains secrecy"}]}'
    parsed = parse_llm_json(payload)
    assert parsed["categories"][0]["category"] == "Confidentiality"
    assert parsed["categories"][0]["category_index"] == 1


def test_parse_with_extra_text():
    payload = "Here you go:\n{ \"categories\": [{\"category\": \"NONE\", \"category_index\": 0, \"confidence\": 0.4, \"reason\": \"No match\" }]}"
    parsed = parse_llm_json(payload)
    assert parsed["categories"][0]["category"] == "NONE"
    assert parsed["categories"][0]["category_index"] == 0


def test_parse_failure_returns_diagnostics():
    payload = "No JSON here"
    parsed = parse_llm_json(payload)
    assert parsed["error"] == "PARSING_ERROR"
    assert parsed["categories"] == []
    assert parsed["raw_output"] == payload
