import textwrap

import pytest

from contract_cuad.chunking import chunk_contract_into_clauses


SAMPLE_CONTRACT = textwrap.dedent(
    """
    Either party may terminate this Agreement for convenience upon thirty (30) days’ prior written notice to the other party, without cause and without any further liability, except for payment of fees accrued up to the effective date of termination.

    In the event of a Change of Control of Customer, Customer shall provide written notice to Provider within ten (10) days. Provider may, at its sole discretion, terminate this Agreement upon thirty (30) days’ prior written notice following such Change of Control.

    Neither this Agreement nor any of the rights or obligations hereunder may be assigned or transferred by Customer, whether by operation of law or otherwise, without the prior written consent of Provider. Any attempted assignment in violation of this Section shall be null and void.

    Except for liability arising from gross negligence, willful misconduct, or Customer’s breach of its payment obligations, each party’s aggregate liability under this Agreement shall in no event exceed the total amounts paid by Customer to Provider during the twelve (12) months immediately preceding the event giving rise to the claim.

    In no event shall Provider be liable to Customer for any indirect, incidental, consequential, special, punitive, or exemplary damages, including loss of profits or business interruption, even if advised of the possibility of such damages.

    During the term of this Agreement and for a period of twelve (12) months thereafter, neither party shall, directly or indirectly, solicit for employment or hire any employee of the other party who was involved in the performance of this Agreement, without the prior written consent of such other party.
    """
).strip()


class DummyTokenizer:
    def __call__(self, text, add_special_tokens=False, padding=False, truncation=False):
        tokens = text.split()
        return {"input_ids": [0] * len(tokens)}


def test_chunk_contract_into_clauses_splits_by_paragraphs():
    tokenizer = DummyTokenizer()
    clauses = chunk_contract_into_clauses(
        SAMPLE_CONTRACT,
        tokenizer=tokenizer,
        max_tokens=300,
        paragraph_overlap=0,
    )
    assert len(clauses) == 6
    assert "Either party may terminate this Agreement for convenience" in clauses[0]
    assert "In the event of a Change of Control of Customer" in clauses[1]
    assert "Neither this Agreement nor any of the rights or obligations hereunder may be assigned" in clauses[2]


def test_chunk_contract_into_clauses_splits_long_paragraph():
    tokenizer = DummyTokenizer()
    long_paragraph = " ".join(["This sentence makes the paragraph longer."] * 120)
    clauses = chunk_contract_into_clauses(long_paragraph, tokenizer, max_tokens=40)
    assert len(clauses) > 1
    assert all(len(clause.split()) <= 40 for clause in clauses)
