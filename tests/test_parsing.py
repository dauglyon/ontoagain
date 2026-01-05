"""Tests for XML parsing from IDENTIFY output."""

import pytest

from ontoagain.identify import parse_xml_output
from ontoagain.xml_utils import strip_tags, clean_llm_xml


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
