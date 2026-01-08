"""Tests for the RELATE phase."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from ontoagain.models import RawRelationship, Relationship
from ontoagain.relate import (
    extract_matched_concepts_from_xml,
    relate_extract,
    relate_disambiguate,
    _parse_relation_xml,
    parse_raw_relationships_from_xml,
)


class TestExtractMatchedConcepts:
    """Tests for extracting matched concepts from XML."""

    def test_simple_concept_with_match(self):
        """Test extraction of a concept with a single match."""
        xml = '''<concept context="clonidine; antihypertensive">clonidine
  <match ontology="MESH" id="MESH:D003000" label="clonidine"/>
</concept>'''
        concepts = extract_matched_concepts_from_xml(xml)
        assert len(concepts) == 1
        assert concepts[0]["text"] == "clonidine"
        assert concepts[0]["context"] == "clonidine; antihypertensive"
        assert len(concepts[0]["matches"]) == 1
        assert concepts[0]["matches"][0]["id"] == "MESH:D003000"

    def test_concept_with_multiple_matches(self):
        """Test extraction of a concept with multiple matches."""
        xml = '''<concept context="p53; tumor protein">p53
  <match ontology="PR" id="PR:000011233" label="cellular tumor antigen p53"/>
  <match ontology="HGNC" id="HGNC:11998" label="TP53"/>
</concept>'''
        concepts = extract_matched_concepts_from_xml(xml)
        assert len(concepts) == 1
        assert len(concepts[0]["matches"]) == 2

    def test_multiple_concepts(self):
        """Test extraction of multiple concepts."""
        xml = '''text before <concept context="drug">clonidine
  <match ontology="MESH" id="MESH:D003000" label="clonidine"/>
</concept> causes <concept context="disease">hypertension
  <match ontology="MESH" id="MESH:D006973" label="hypertension"/>
</concept> text after'''
        concepts = extract_matched_concepts_from_xml(xml)
        assert len(concepts) == 2
        assert concepts[0]["text"] == "clonidine"
        assert concepts[1]["text"] == "hypertension"

    def test_concept_without_match(self):
        """Test extraction of a concept with no match."""
        xml = '''<concept context="something">term</concept>'''
        concepts = extract_matched_concepts_from_xml(xml)
        assert len(concepts) == 1
        assert concepts[0]["text"] == "term"
        assert concepts[0]["matches"] == []


class TestRelateExtract:
    """Tests for relationship extraction."""

    def test_extract_basic(self):
        """Test basic relationship extraction with mocked LLM."""
        xml = '''<Documents><D id="test"><C n="1" q="drug">clonidine<M id="MESH:D003000" o="MESH" l="clonidine"/></C> causes <C n="2" q="disease">hypertension<M id="MESH:D006973" o="MESH" l="hypertension"/></C></D></Documents>'''

        mock_response = '''<relations>
<R s="MESH:D003000" o="MESH:D006973" p="causes" e="1,2">clonidine causes hypertension</R>
</relations>'''

        with patch("ontoagain.relate.call_llm_async", return_value=mock_response):
            relationships = relate_extract(xml, model="test-model", verbose=False)

        assert len(relationships) == 1
        assert relationships[0].doc_id == "test"
        assert relationships[0].subject_id == "MESH:D003000"
        assert relationships[0].object_id == "MESH:D006973"
        assert relationships[0].predicate == "causes"
        assert relationships[0].concept_ids == ["1", "2"]
        assert relationships[0].summary == "clonidine causes hypertension"

    def test_extract_multiple_relationships(self):
        """Test extraction of multiple relationships."""
        xml = '''<Documents><D id="test"><C n="1" q="A">A<M id="A:1"/></C>
<C n="2" q="B">B<M id="B:1"/></C>
<C n="3" q="C">C<M id="C:1"/></C></D></Documents>'''

        mock_response = '''<relations>
<R s="A:1" o="B:1" p="activates" e="1,2">A activates B</R>
<R s="B:1" o="C:1" p="inhibits" e="2,3">B inhibits C</R>
</relations>'''

        with patch("ontoagain.relate.call_llm_async", return_value=mock_response):
            relationships = relate_extract(xml, model="test-model", verbose=False)

        assert len(relationships) == 2
        assert relationships[0].predicate == "activates"
        assert relationships[1].predicate == "inhibits"

    def test_extract_no_relationships(self):
        """Test extraction when no relationships found."""
        xml = '''<Documents><D id="test"><C n="1" q="A">A<M id="A:1"/></C></D></Documents>'''

        mock_response = '<relations></relations>'

        with patch("ontoagain.relate.call_llm_async", return_value=mock_response):
            relationships = relate_extract(xml, model="test-model", verbose=False)

        assert len(relationships) == 0

    def test_extract_handles_malformed_response(self):
        """Test graceful handling of malformed LLM response."""
        xml = '''<Documents><D id="test"><C n="1" q="A">A<M id="A:1"/></C></D></Documents>'''

        mock_response = "not valid xml"

        with patch("ontoagain.relate.call_llm_async", return_value=mock_response):
            relationships = relate_extract(xml, model="test-model", verbose=False)

        assert len(relationships) == 0


class TestRelateDisambiguate:
    """Tests for relationship predicate disambiguation."""

    @pytest.mark.asyncio
    async def test_disambiguate_basic(self):
        """Test basic predicate disambiguation."""
        from ontoagain.relate import _disambiguate_batch_async

        relationships = [
            RawRelationship(
                doc_id="test",
                subject_id="A:1",
                object_id="B:1",
                predicate="causes",
                concept_ids=["1", "2"],
                summary="A causes B",
            )
        ]

        candidates_by_predicate = {
            "causes": [
                {"id": "RO:0002410", "ontology": "RO", "label": "causally_related_to", "definition": "", "synonyms": ["causes"]},
                {"id": "RO:0002411", "ontology": "RO", "label": "some_other", "definition": "", "synonyms": []},
            ]
        }

        mock_response = '{"0": "RO:0002410"}'

        async def mock_llm(*args, **kwargs):
            return mock_response

        with patch("ontoagain.relate.call_llm_async", side_effect=mock_llm):
            results = await _disambiguate_batch_async(
                relationships, candidates_by_predicate, "test-model", 1, 1, None
            )

        assert len(results) == 1
        assert results[0].predicate_id == "RO:0002410"
        assert results[0].predicate_label == "causally_related_to"
        assert results[0].predicate_raw == "causes"
        assert results[0].concept_ids == ["1", "2"]
        assert results[0].summary == "A causes B"

    @pytest.mark.asyncio
    async def test_disambiguate_no_match(self):
        """Test disambiguation when no candidate selected."""
        from ontoagain.relate import _disambiguate_batch_async

        relationships = [
            RawRelationship(
                doc_id="test",
                subject_id="A:1",
                object_id="B:1",
                predicate="weird_relation",
                concept_ids=["1"],
                summary="A weird_relation B",
            )
        ]

        candidates_by_predicate = {"weird_relation": []}

        results = await _disambiguate_batch_async(
            relationships, candidates_by_predicate, "test-model", 1, 1, None
        )

        assert len(results) == 1
        assert results[0].predicate_id == ""
        assert results[0].predicate_label == ""
        assert results[0].predicate_raw == "weird_relation"
        assert results[0].concept_ids == ["1"]


class TestParseRawRelationshipsFromXml:
    """Tests for parsing raw relationships from XML."""

    def test_parse_single_relationship(self):
        """Test parsing a single relationship from XML."""
        xml = '''<relations>
<R d="doc1" s="MESH:D003000" o="MESH:D006973" p="causes" e="1,2">clonidine causes hypertension</R>
</relations>'''
        relationships = parse_raw_relationships_from_xml(xml)

        assert len(relationships) == 1
        assert relationships[0].doc_id == "doc1"
        assert relationships[0].subject_id == "MESH:D003000"
        assert relationships[0].object_id == "MESH:D006973"
        assert relationships[0].predicate == "causes"
        assert relationships[0].concept_ids == ["1", "2"]
        assert relationships[0].summary == "clonidine causes hypertension"

    def test_parse_multiple_relationships(self):
        """Test parsing multiple relationships from XML."""
        xml = '''<relations>
<R d="doc1" s="A:1" o="B:1" p="activates" e="1,2">A activates B</R>
<R d="doc2" s="C:1" o="D:1" p="inhibits" e="3,4">C inhibits D</R>
</relations>'''
        relationships = parse_raw_relationships_from_xml(xml)

        assert len(relationships) == 2
        assert relationships[0].doc_id == "doc1"
        assert relationships[0].predicate == "activates"
        assert relationships[1].doc_id == "doc2"
        assert relationships[1].predicate == "inhibits"

    def test_parse_empty_relations(self):
        """Test parsing empty relations XML."""
        xml = '<relations></relations>'
        relationships = parse_raw_relationships_from_xml(xml)
        assert len(relationships) == 0

    def test_parse_missing_attributes(self):
        """Test parsing with missing optional attributes."""
        xml = '''<relations>
<R s="A:1" o="B:1" p="related">summary</R>
</relations>'''
        relationships = parse_raw_relationships_from_xml(xml)

        assert len(relationships) == 1
        assert relationships[0].doc_id == ""
        assert relationships[0].concept_ids == []
        assert relationships[0].summary == "summary"
