"""XML parsing utilities with robust error handling for LLM outputs."""

import re
import xml.etree.ElementTree as ET


def fix_truncated_xml(xml_text: str) -> str:
    """Fix truncated XML output from LLM.

    Handles cases where output was cut off mid-tag due to token limits.
    Supports both <C q="..."> (compact) and <concept context="..."> (legacy) formats.
    """
    text = xml_text.rstrip()

    # Check for unclosed <C> tag (compact format)
    last_open_c = text.rfind("<C ")
    if last_open_c != -1:
        last_close_c = text.rfind("</C>")
        if last_close_c < last_open_c:
            remaining = text[last_open_c:]
            if ">" not in remaining:
                text = text + '">TRUNCATED</C>'
            else:
                text = text + "</C>"

    # Check for unclosed <concept> tag (legacy format)
    last_open = text.rfind("<concept")
    if last_open != -1:
        last_close = text.rfind("</concept>")
        if last_close < last_open:
            remaining = text[last_open:]
            if ">" not in remaining:
                text = text + '">TRUNCATED</concept>'
            else:
                text = text + "</concept>"

    return text


def clean_llm_xml(xml_text: str) -> str:
    """Clean XML output from LLMs for parsing.

    Handles common issues:
    - XML declarations (<?xml ...?>)
    - Markdown code block wrappers (```xml ... ```)
    - Unescaped ampersands
    - Unescaped less-than/greater-than in text content (e.g., p < 0.05)

    Args:
        xml_text: Raw XML text from LLM

    Returns:
        Cleaned XML text ready for parsing
    """
    text = xml_text

    # Strip XML declaration (must be at start of document, causes issues when we wrap)
    text = re.sub(r"^\s*<\?xml[^?]*\?>\s*", "", text)

    # Strip markdown code block wrappers
    text = re.sub(r"^```(?:xml)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)

    # Escape unescaped ampersands (but not already-escaped entities)
    text = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)", "&amp;", text)

    # Escape < and > that appear in text content (not tags)
    # This handles scientific notation like "p < 0.05" or "n > 100"
    # Match < or > that are:
    # - preceded by a space/digit and followed by a space/digit (comparison operators)
    # - not part of a tag (not followed by tag name pattern)
    text = _escape_comparison_operators(text)

    return text


def _escape_comparison_operators(text: str) -> str:
    """Escape < and > when used as comparison operators, not XML tags.

    Heuristic: if < is followed by a space or digit, it's likely a comparison.
    If < is followed by a letter or /, it's likely an XML tag.
    """
    result = []
    i = 0
    while i < len(text):
        char = text[i]

        if char == '<':
            # Look ahead to determine if this is a tag or comparison
            rest = text[i + 1:i + 20] if i + 1 < len(text) else ""

            # It's a tag if followed by: letter, /, !, or ?
            if rest and re.match(r'^[a-zA-Z/?!]', rest):
                result.append(char)
            else:
                # It's a comparison operator - escape it
                result.append('&lt;')
        elif char == '>':
            # > is trickier - check if we're likely closing a tag
            # Look back to see if we recently opened a tag
            recent = ''.join(result[-50:]) if len(result) >= 50 else ''.join(result)

            # If there's an unclosed < that looks like a tag start, this closes it
            if re.search(r'<[a-zA-Z/?!][^>]*$', recent):
                result.append(char)
            else:
                # Look at context - if preceded by space/digit, likely comparison
                prev_char = result[-1] if result else ''
                if prev_char in ' \t0123456789':
                    result.append('&gt;')
                else:
                    result.append(char)
        else:
            result.append(char)

        i += 1

    return ''.join(result)


def parse_xml_fragment(xml_text: str, root_tag: str = "root") -> ET.Element:
    """Parse an XML fragment, wrapping in a root element if needed.

    Args:
        xml_text: XML text (may be a fragment without root)
        root_tag: Tag name to use for wrapper element

    Returns:
        Parsed ElementTree Element

    Raises:
        ValueError: If XML cannot be parsed
    """
    cleaned = clean_llm_xml(xml_text)
    wrapped = f"<{root_tag}>{cleaned}</{root_tag}>"

    try:
        return ET.fromstring(wrapped)
    except ET.ParseError as e:
        # Include helpful context in error
        line_no = getattr(e, 'position', (0, 0))[0]
        lines = wrapped.split('\n')
        context_start = max(0, line_no - 3)
        context_end = min(len(lines), line_no + 2)
        context = '\n'.join(f"  {i+1}: {lines[i]}" for i in range(context_start, context_end))
        raise ValueError(f"Failed to parse XML at line {line_no}:\n{context}\n\nError: {e}")


def extract_xml_block(text: str, tag: str) -> str | None:
    """Extract content of a specific XML block from text.

    Args:
        text: Text that may contain XML
        tag: Tag name to extract (e.g., "recommendations")

    Returns:
        Content between opening and closing tags, or None if not found
    """
    pattern = rf"<{tag}[^>]*>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def strip_tags(xml_text: str) -> str:
    """Remove all XML tags, leaving just the text content."""
    return re.sub(r"<[^>]+>", "", xml_text)


def get_element_text(elem: ET.Element) -> str:
    """Get all text content from an element, including nested elements."""
    texts = []
    if elem.text:
        texts.append(elem.text)
    for child in elem:
        texts.append(get_element_text(child))
        if child.tail:
            texts.append(child.tail)
    return " ".join(t for t in texts if t)


def safe_get_text(elem: ET.Element | None) -> str:
    """Safely get text from an element that may be None."""
    if elem is None or elem.text is None:
        return ""
    return elem.text.strip()


def extract_concepts_from_xml(xml_text: str) -> list[dict]:
    """Extract concept information from XML with concept tags.

    Supports both <C n="..." q="..."> (compact) and <concept context="..."> (legacy) formats.

    Args:
        xml_text: XML text with concept tags

    Returns:
        List of dicts with keys: text, context, index, n (concept ID if present)
        Index is the position in the list (for matching back after disambiguation)
    """
    root = parse_xml_fragment(xml_text)
    concepts = []

    # Support both <C n="..." q="..."> (compact) and <concept context="..."> (legacy)
    for elem in root.findall(".//C") + root.findall(".//concept"):
        concepts.append({
            "index": len(concepts),
            "n": elem.get("n", ""),  # Concept ID
            "text": elem.text or "",
            "context": elem.get("q", "") or elem.get("context", ""),
        })

    return concepts


def update_xml_with_matches(
    xml_text: str,
    matches: list[list[dict]],
) -> str:
    """Update concept tags in XML with nested match elements.

    Supports both <C n="..." q="..."> (compact) and <concept context="..."> formats.
    Output uses compact <C n="..." q="..."><M .../></C> format.

    Args:
        xml_text: Original XML with concept tags
        matches: List of matches for each concept (parallel to concept order in XML).
                 Each match is a dict with ontology, id, label.

    Returns:
        Updated XML with <M/> elements nested inside concept tags
    """
    result = xml_text
    offset = 0  # Track offset as we modify the string
    match_idx = 0

    # Pattern to match both formats, capturing n attribute if present
    # <C n="1" q="...">text</C> or <C q="...">text</C> or <concept context="...">text</concept>
    pattern = r'<(C|concept)\s+(?:n="(\d+)"\s+)?(?:q|context)="([^"]*)"?>([^<]*)</(C|concept)>'

    for match in re.finditer(pattern, xml_text):
        if match_idx >= len(matches):
            break

        concept_matches = matches[match_idx]
        match_idx += 1

        # Get n, context, and concept text
        n_attr = match.group(2)  # May be None
        context = match.group(3)
        concept_text = match.group(4)

        # Build nested match elements (compact format: <M id="..." o="..." l="..."/>)
        match_elements = []
        for m in concept_matches:
            match_elements.append(
                f'<M id="{m["id"]}" o="{m["ontology"]}" l="{_escape_attr(m["label"])}"/>'
            )

        # Build new tag (compact format)
        n_part = f'n="{n_attr}" ' if n_attr else ""
        if match_elements:
            matches_xml = "".join(match_elements)
            new_tag = f'<C {n_part}q="{_escape_attr(context)}">{concept_text}{matches_xml}</C>'
        else:
            new_tag = f'<C {n_part}q="{_escape_attr(context)}">{concept_text}</C>'

        # Replace in result (accounting for offset from previous replacements)
        start = match.start() + offset
        end = match.end() + offset
        result = result[:start] + new_tag + result[end:]

        # Update offset
        offset += len(new_tag) - (match.end() - match.start())

    return result


def _escape_attr(value: str) -> str:
    """Escape a string for use as an XML attribute value."""
    return (value
            .replace("&", "&amp;")
            .replace('"', "&quot;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def _escape_text(value: str) -> str:
    """Escape a string for use as XML text content."""
    return (value
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def wrap_document(content: str, doc_id: str) -> str:
    """Wrap content in a single <D> document element."""
    return f'<D id="{_escape_attr(doc_id)}">{content}</D>'


def wrap_documents(doc_contents: list[tuple[str, str]]) -> str:
    """Wrap multiple documents in a <Documents> root element.

    Args:
        doc_contents: List of (doc_id, content) tuples

    Returns:
        XML string with <Documents><D id="...">...</D>...</Documents> structure
    """
    parts = ["<Documents>"]
    for doc_id, content in doc_contents:
        parts.append(wrap_document(content, doc_id))
    parts.append("</Documents>")
    return "\n".join(parts)


def extract_documents(xml_text: str) -> list[tuple[str, str]]:
    """Extract documents from a <Documents> wrapper.

    Args:
        xml_text: XML with <Documents><D id="...">...</D>...</Documents> structure

    Returns:
        List of (doc_id, content) tuples
    """
    root = parse_xml_fragment(xml_text)
    documents = []

    for doc_elem in root.findall(".//D"):
        doc_id = doc_elem.get("id", "")
        # Get inner XML (everything inside the <D> tag)
        inner_parts = []
        if doc_elem.text:
            inner_parts.append(doc_elem.text)
        for child in doc_elem:
            inner_parts.append(ET.tostring(child, encoding="unicode"))
            if child.tail:
                inner_parts.append(child.tail)
        content = "".join(inner_parts)
        documents.append((doc_id, content))

    return documents
