"""Tests for XML parsing from IDENTIFY output."""

import pytest

from ontoagain.identify import parse_xml_output, _strip_tags, _clean_xml


class TestStripTags:
    def test_simple(self):
        xml = "<claim>Hello world</claim>"
        assert _strip_tags(xml) == "Hello world"

    def test_nested(self):
        xml = "<claim>Hello <concept>world</concept></claim>"
        assert _strip_tags(xml) == "Hello world"

    def test_with_attributes(self):
        xml = '<concept context="test">hello</concept>'
        assert _strip_tags(xml) == "hello"


class TestCleanXml:
    def test_unescaped_ampersand(self):
        xml = "R&D department"
        assert _clean_xml(xml) == "R&amp;D department"

    def test_already_escaped(self):
        xml = "R&amp;D department"
        assert _clean_xml(xml) == "R&amp;D department"


class TestParseXmlOutput:
    def test_simple_claim(self):
        xml = '<claim type="finding">Test claim text.</claim>'
        claims, concepts = parse_xml_output(xml)

        assert len(claims) == 1
        assert claims[0].type == "finding"
        assert "Test claim text" in claims[0].text

    def test_claim_with_concept(self):
        xml = '''<claim type="method">
            We studied <concept context="tumor protein p53">p53</concept> expression.
        </claim>'''
        claims, concepts = parse_xml_output(xml)

        assert len(claims) == 1
        assert len(concepts) == 1
        assert concepts[0].text == "p53"
        assert concepts[0].context == "tumor protein p53"

    def test_multiple_claims(self):
        xml = '''
        <claim type="method">Method text here.</claim>
        <claim type="finding">Finding text here.</claim>
        '''
        claims, concepts = parse_xml_output(xml)

        assert len(claims) == 2
        assert claims[0].type == "method"
        assert claims[1].type == "finding"

    def test_multiple_concepts(self):
        xml = '''<claim type="finding">
            <concept context="p53">p53</concept> regulates
            <concept context="apoptosis">apoptosis</concept>.
        </claim>'''
        claims, concepts = parse_xml_output(xml)

        assert len(concepts) == 2
        assert concepts[0].text == "p53"
        assert concepts[1].text == "apoptosis"

    def test_concept_positions(self):
        xml = '''<claim type="finding">
            The <concept context="p53">p53</concept> protein is important.
        </claim>'''
        claims, concepts = parse_xml_output(xml)

        # Position should be found in stripped text
        assert len(concepts) == 1
        assert concepts[0].start >= 0
        assert concepts[0].end > concepts[0].start
