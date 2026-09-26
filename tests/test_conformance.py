"""Run the committed conformance vectors in spec/vectors against IMPSY.

Other implementations run the same files in their own test suites; see
spec/README.md.
"""

import copy
import json
import math

import pytest

from impsy import conformance

VECTOR_DIR = conformance.DEFAULT_VECTOR_DIR
TOLERANCE = 1e-9


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


def assert_matches(actual, expected, path="expected"):
    if isinstance(expected, float) or isinstance(actual, float):
        assert math.isclose(
            actual, expected, rel_tol=0, abs_tol=TOLERANCE
        ), f"{path}: {actual} != {expected}"
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: {actual!r} is not a list"
        assert len(actual) == len(expected), f"{path}: {actual} != {expected}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_matches(a, e, f"{path}[{i}]")
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: {actual!r} is not a dict"
        assert actual.keys() == expected.keys(), f"{path}: {actual} != {expected}"
        for key in expected:
            assert_matches(actual[key], expected[key], f"{path}.{key}")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def test_vector_files_present():
    assert set(DOCUMENTS) == set(conformance.VECTOR_FILES)


@pytest.mark.parametrize("filename,case", CASES)
def test_conformance_case(filename, case):
    inputs = {k: v for k, v in copy.deepcopy(case).items() if k != "expected"}
    actual = conformance.RUNNERS[filename](inputs)
    assert_matches(actual, case["expected"])


def test_vectors_are_current():
    """Committed vectors must match what the generator produces.

    If this fails after an intentional behaviour change, regenerate with
    `poetry run python -m impsy.conformance` and bump SPEC_VERSION.
    """
    generated = conformance.build_vectors()
    for filename, document in generated.items():
        committed = (VECTOR_DIR / filename).read_text()
        assert committed == conformance.dump(document), f"{filename} is out of date"
