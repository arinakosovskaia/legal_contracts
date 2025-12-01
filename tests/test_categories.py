from pathlib import Path

from contract_cuad.categories import build_category_lookup, ensure_category_name, load_cuad_categories


def test_load_cuad_categories_local_csv(tmp_path: Path):
    csv_path = tmp_path / "category_descriptions.csv"
    csv_path.write_text("Category,Description\nConfidentiality,Keep data secret\nAssignment,Control transfers\n")
    categories = load_cuad_categories(str(csv_path))
    assert len(categories) == 2
    assert categories[0].index == 1
    assert categories[0].name == "Confidentiality"


def test_ensure_category_name_case_insensitive(tmp_path: Path):
    csv_path = tmp_path / "cats.csv"
    csv_path.write_text("Category,Description\nIndemnity,Risk sharing\n")
    categories = load_cuad_categories(str(csv_path))
    lookup = build_category_lookup(categories)
    assert "indemnity" in lookup
    category = ensure_category_name(categories, "INDEMNITY")
    assert category.description == "Risk sharing"
