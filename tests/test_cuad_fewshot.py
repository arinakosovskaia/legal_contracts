from contract_cuad.cuad_fewshot import _match_question_to_category
from contract_cuad.categories import Category


def test_match_question_with_category_prefix_and_dash():
    cats = [
        Category(index=1, name="Confidentiality", description=""),
        Category(index=2, name="Anti-Assignment", description=""),
    ]
    question = "Category: Confidentiality – Does the contract contain any confidentiality obligations?"
    matched = _match_question_to_category(question, cats)
    assert matched is not None
    assert matched.name == "Confidentiality"


def test_match_question_with_simple_name_only():
    cats = [Category(index=1, name="Change of Control", description="")]
    question = "Change of Control"
    matched = _match_question_to_category(question, cats)
    assert matched is not None
    assert matched.name == "Change of Control"


def test_match_question_with_small_typo_uses_fuzzy_match():
    cats = [Category(index=1, name="Anti-Assignment", description="")]
    question = "Category: Anti Assignment - Are assignments restricted?"
    matched = _match_question_to_category(question, cats)
    assert matched is not None
    assert matched.name == "Anti-Assignment"
