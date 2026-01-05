"""Tests for ontology indexing."""

import pytest
from pathlib import Path

from ontoagain.models import OntologyTerm
from ontoagain.index import create_embedding_text, load_ontology


class TestCreateEmbeddingText:
    def test_label_only(self):
        term = OntologyTerm(
            id="GO:0001",
            ontology="GO",
            label="test process",
            synonyms=[],
            definition="",
        )
        text = create_embedding_text(term)
        assert text == "test process"

    def test_with_synonyms(self):
        term = OntologyTerm(
            id="GO:0001",
            ontology="GO",
            label="test process",
            synonyms=["synonym1", "synonym2"],
            definition="",
        )
        text = create_embedding_text(term)
        assert "test process" in text
        assert "synonym1" in text
        assert "synonym2" in text

    def test_with_definition(self):
        term = OntologyTerm(
            id="GO:0001",
            ontology="GO",
            label="test process",
            synonyms=[],
            definition="A biological process for testing.",
        )
        text = create_embedding_text(term)
        assert "test process" in text
        assert "biological process" in text

    def test_full_term(self):
        term = OntologyTerm(
            id="GO:0001",
            ontology="GO",
            label="apoptotic process",
            synonyms=["programmed cell death", "apoptosis"],
            definition="A form of cell death.",
        )
        text = create_embedding_text(term)
        assert "apoptotic process" in text
        assert "programmed cell death" in text
        assert "apoptosis" in text
        assert "cell death" in text


class TestLoadOntology:
    def test_nonexistent_file(self):
        with pytest.raises(Exception):
            load_ontology(Path("/nonexistent/file.obo"))
