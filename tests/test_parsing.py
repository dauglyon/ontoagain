"""Tests for XML parsing from IDENTIFY output."""

import pytest

from ontoagain.identify import parse_xml_output
from ontoagain.xml_utils import (
    strip_tags,
    clean_llm_xml,
    fix_truncated_xml,
    extract_concepts_from_xml,
    update_xml_with_matches,
)


class TestStripTags:
    def test_simple(self):
        xml = "<concept>Hello world</concept>"
        assert strip_tags(xml) == "Hello world"

    def test_nested(self):
        xml = "Hello <concept>world</concept>"
        assert strip_tags(xml) == "Hello world"

    def test_with_attributes(self):
        xml = '<concept context="test">hello</concept>'
        assert strip_tags(xml) == "hello"


class TestCleanXml:
    def test_unescaped_ampersand(self):
        xml = "R&D department"
        assert clean_llm_xml(xml) == "R&amp;D department"

    def test_already_escaped(self):
        xml = "R&amp;D department"
        assert clean_llm_xml(xml) == "R&amp;D department"


class TestParseXmlOutput:
    def test_simple_concept(self):
        xml = '<concept context="tumor protein p53">p53</concept>'
        concepts = parse_xml_output(xml)

        assert len(concepts) == 1
        assert concepts[0].text == "p53"
        assert concepts[0].context == "tumor protein p53"

    def test_multiple_concepts(self):
        xml = '''
            <concept context="p53; tumor protein p53">p53</concept> regulates
            <concept context="apoptosis; programmed cell death">apoptosis</concept>.
        '''
        concepts = parse_xml_output(xml)

        assert len(concepts) == 2
        assert concepts[0].text == "p53"
        assert concepts[1].text == "apoptosis"

    def test_concept_positions(self):
        xml = 'The <concept context="p53">p53</concept> protein is important.'
        concepts = parse_xml_output(xml)

        # Position should be found in stripped text
        assert len(concepts) == 1
        assert concepts[0].start >= 0
        assert concepts[0].end > concepts[0].start

    def test_concept_with_semicolon_context(self):
        xml = '<concept context="6mA; N6-methyladenine; adenine methylation">6mA</concept>'
        concepts = parse_xml_output(xml)

        assert len(concepts) == 1
        assert concepts[0].text == "6mA"
        assert "N6-methyladenine" in concepts[0].context

    def test_compact_format(self):
        """Test parsing compact <C q="..."> format."""
        xml = '<C q="p53; TP53; tumor protein p53">p53</C>'
        concepts = parse_xml_output(xml)

        assert len(concepts) == 1
        assert concepts[0].text == "p53"
        assert concepts[0].context == "p53; TP53; tumor protein p53"

    def test_compact_format_multiple(self):
        """Test parsing multiple compact tags."""
        xml = '''The <C q="p53; TP53">p53</C> protein regulates <C q="apoptosis; PCD">apoptosis</C>.'''
        concepts = parse_xml_output(xml)

        assert len(concepts) == 2
        assert concepts[0].text == "p53"
        assert concepts[1].text == "apoptosis"


class TestCompactFormat:
    """Tests for compact XML format (<C q="...">)."""

    def test_strip_tags_compact(self):
        xml = '<C q="test">hello</C>'
        assert strip_tags(xml) == "hello"

    def test_fix_truncated_compact(self):
        """Test fixing truncated compact tags."""
        # Truncated in attribute
        xml = 'Text <C q="partial'
        fixed = fix_truncated_xml(xml)
        assert "</C>" in fixed

        # Truncated after attribute
        xml = 'Text <C q="complete">partial'
        fixed = fix_truncated_xml(xml)
        assert fixed.endswith("</C>")

    def test_extract_concepts_compact(self):
        """Test extracting concepts from compact format."""
        xml = '<C q="p53; TP53">p53</C> and <C q="apoptosis">apoptosis</C>'
        concepts = extract_concepts_from_xml(xml)

        assert len(concepts) == 2
        assert concepts[0]["text"] == "p53"
        assert concepts[0]["context"] == "p53; TP53"
        assert concepts[1]["text"] == "apoptosis"

    def test_extract_concepts_mixed_format(self):
        """Test extracting from mixed compact and legacy formats."""
        xml = '<C q="p53">p53</C> and <concept context="apoptosis">apoptosis</concept>'
        concepts = extract_concepts_from_xml(xml)

        assert len(concepts) == 2
        assert concepts[0]["context"] == "p53"
        assert concepts[1]["context"] == "apoptosis"

    def test_update_xml_compact_to_readable(self):
        """Test that compact input is converted to readable output."""
        xml = '<C q="p53; TP53">p53</C>'
        matches = [[{"ontology": "PR", "id": "PR:000001", "label": "p53 protein"}]]

        result = update_xml_with_matches(xml, matches)

        # Output should use readable <concept> format
        assert "<concept context=" in result
        assert "<C " not in result
        assert 'id="PR:000001"' in result

    def test_update_xml_no_matches(self):
        """Test conversion with no matches."""
        xml = '<C q="unknown">unknown term</C>'
        matches = [[]]

        result = update_xml_with_matches(xml, matches)

        assert "<concept context=" in result
        assert "<match" not in result
