"""Run the committed conformance vectors in spec/vectors against IMPSY.

Other implementations run the same files in their own test suites; see
spec/README.md.
"""

import copy
import json

import pytest

from impsy import conformance

VECTOR_DIR = conformance.DEFAULT_VECTOR_DIR


def load_documents():
    return {
        path.name: json.loads(path.read_text())
        for path in sorted(VECTOR_DIR.glob("*.json"))
    }


DOCUMENTS = load_documents()
CASES = [
    pytest.param(filename, case, id=f"{filename[:-5]}::{case['name']}")
    for filename, document in DOCUMENTS.items()
    for case in document["cases"]
]


def test_vector_files_present():
    assert set(DOCUMENTS) == set(conformance.VECTOR_FILES)


@pytest.mark.parametrize("filename,case", CASES)
def test_conformance_case(filename, case):
    inputs = {k: v for k, v in copy.deepcopy(case).items() if k != "expected"}
    actual = conformance.RUNNERS[filename](inputs)
    tolerance = DOCUMENTS[filename]["tolerance"]
    found = conformance.mismatch(actual, case["expected"], tolerance)
    assert found is None, found


def test_vectors_are_current():
    """Committed vectors must match what the generator produces.

    If this fails after an intentional behaviour change, regenerate with
    `poetry run python -m impsy.conformance` and bump SPEC_VERSION.
    """
    for filename, document in conformance.build_vectors().items():
        found = conformance.mismatch(
            document, DOCUMENTS[filename], document["tolerance"]
        )
        assert found is None, f"{filename} is out of date: {found}"
