"""IDENTIFY agent - extract concepts from scientific text."""

import json
import re
from pathlib import Path

from ontoagain.llm import call_llm, get_client
from ontoagain.models import Concept, OntologyMetadata
from ontoagain.xml_utils import (
    fix_truncated_xml,
    get_element_text,
    parse_xml_fragment,
    strip_tags,
)

# Load prompt templates
PROMPT_PATH = Path(__file__).parent / "prompts" / "identify.txt"
CONTEXT_PROMPT_PATH = Path(__file__).parent / "prompts" / "extract_context.txt"

# Default model for all modules
DEFAULT_MODEL = "anthropic/claude-sonnet"

# Chunking parameters
MAX_CHUNK_CHARS = 20000  # Target chunk size
CHUNK_OVERLAP_CHARS = 500  # Overlap between chunks for context continuity
CONTEXT_EXTRACTION_CHARS = 25000  # Only use first N chars for context extraction


def load_prompt() -> str:
    """Load the identify prompt template."""
    return PROMPT_PATH.read_text()


def load_context_prompt() -> str:
    """Load the context extraction prompt template."""
    return CONTEXT_PROMPT_PATH.read_text()


def extract_document_context(
    text: str,
    model: str,
    verbose: bool = False,
) -> dict:
    """Extract document-level context (abbreviations, topic, key entities).

    Args:
        text: Full paper text
        model: LLM model to use
        verbose: Print debug info

    Returns:
        Dict with main_topic, abbreviations, key_entities
    """
    prompt_template = load_context_prompt()
    prompt = prompt_template.replace("{text}", text)

    client, model_name = get_client(model)

    if verbose:
        print("Extracting document context...")

    messages = [{"role": "user", "content": prompt}]
    result = call_llm(client, model_name, messages)

    if verbose:
        print(f"Context extraction complete. Response length: {len(result) if result else 0} chars")

    # Parse JSON response
    try:
        # Try to extract JSON from response
        match = re.search(r"\{.*\}", result, re.DOTALL)
        if match:
            context = json.loads(match.group())
        else:
            context = {"main_topic": "", "abbreviations": {}, "key_entities": []}
    except (json.JSONDecodeError, TypeError):
        context = {"main_topic": "", "abbreviations": {}, "key_entities": []}

    return context


def format_document_context(context: dict) -> str:
    """Format extracted context for injection into identify prompt.

    Args:
        context: Dict from extract_document_context

    Returns:
        Formatted string for prompt injection
    """
    if not context or not any(context.values()):
        return ""

    lines = ["## Document Context", ""]

    if context.get("main_topic"):
        lines.append(f"**This paper is about**: {context['main_topic']}")
        lines.append("")

    if context.get("abbreviations"):
        lines.append("**Abbreviations used in this paper**:")
        for abbrev, expansion in context["abbreviations"].items():
            lines.append(f"- {abbrev} = {expansion}")
        lines.append("")

    if context.get("key_entities"):
        entities = ", ".join(context["key_entities"][:50])  # Limit to 50
        lines.append(f"**Key entities in this paper**: {entities}")
        lines.append("")

    return "\n".join(lines)


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Split text into chunks at paragraph boundaries.

    Args:
        text: Text to chunk
        max_chars: Target maximum characters per chunk
        overlap: Characters of overlap between chunks

    Returns:
        List of text chunks
    """
    # Split on double newlines (paragraphs)
    paragraphs = re.split(r"\n\n+", text)

    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para) + 2  # +2 for newlines

        if current_size + para_size > max_chars and current_chunk:
            # Save current chunk
            chunks.append("\n\n".join(current_chunk))

            # Start new chunk with overlap
            # Include last paragraph(s) from previous chunk for context
            overlap_paras = []
            overlap_size = 0
            for p in reversed(current_chunk):
                if overlap_size + len(p) > overlap:
                    break
                overlap_paras.insert(0, p)
                overlap_size += len(p) + 2

            current_chunk = overlap_paras
            current_size = overlap_size

        current_chunk.append(para)
        current_size += para_size

    # Don't forget the last chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def format_ontology_info(metadata: list[OntologyMetadata]) -> str:
    """Format ontology metadata for injection into the prompt.

    Args:
        metadata: List of ontology metadata

    Returns:
        Formatted string describing available ontologies
    """
    if not metadata:
        return ""

    lines = ["## Available Ontologies", ""]
    lines.append("Generate search terms that match these ontology term formats:")
    lines.append("")

    for m in metadata:
        lines.append(f"- **{m.name}**: {m.description}")
        lines.append(f"  - Term format: {m.term_format}")

    lines.append("")
    return "\n".join(lines)


def identify_chunk(
    text: str,
    model: str,
    ontology_info: str,
    document_context: str,
    verbose: bool = False,
    chunk_num: int = 0,
    total_chunks: int = 1,
) -> str:
    """Process a single chunk of text.

    Args:
        text: Text chunk to process
        model: LLM model to use
        ontology_info: Formatted ontology info string
        document_context: Formatted document context string
        verbose: Print debug info
        chunk_num: Current chunk number (1-indexed)
        total_chunks: Total number of chunks

    Returns:
        XML-tagged text
    """
    prompt_template = load_prompt()

    # Build prompt with context
    prompt = prompt_template.replace("{ontology_info}", ontology_info)
    prompt = prompt.replace("{document_context}", document_context)
    prompt = prompt.replace("{text}", text)

    client, model_name = get_client(model)

    if verbose:
        chunk_info = f"[{chunk_num}/{total_chunks}] " if total_chunks > 1 else ""
        print(f"  {chunk_info}Processing {len(text)} chars...", flush=True)

    messages = [{"role": "user", "content": prompt}]
    result = call_llm(client, model_name, messages)

    if verbose:
        print(f"    -> {len(result) if result else 0} chars output", flush=True)

    # Fix any truncated XML
    result = fix_truncated_xml(result)

    return result


def combine_chunk_outputs(chunks: list[str], overlap: int = CHUNK_OVERLAP_CHARS) -> str:
    """Combine tagged chunk outputs, handling overlapping regions.

    The overlap regions may have duplicate tags. We keep the tags from the
    first chunk (which had more preceding context).

    Args:
        chunks: List of tagged text chunks
        overlap: Approximate overlap size used during chunking

    Returns:
        Combined tagged text
    """
    if not chunks:
        return ""
    if len(chunks) == 1:
        return chunks[0]

    # Simple approach: find the overlap region and remove it from subsequent chunks
    # The overlap was at paragraph boundaries, so look for matching text
    combined = chunks[0]

    for i in range(1, len(chunks)):
        chunk = chunks[i]

        # Find where the overlap ends in this chunk
        # Look for the first paragraph that wasn't in the previous chunk
        # Simple heuristic: skip the first ~overlap characters worth of content

        # Find a good split point - look for a paragraph break after overlap region
        # Strip any leading text that was in the previous chunk

        # Find common suffix of combined / prefix of chunk
        # Use a simpler approach: find first double-newline after overlap chars
        split_pos = chunk.find("\n\n", overlap)
        if split_pos != -1:
            # Skip the overlap region
            chunk = chunk[split_pos + 2:]

        combined += "\n\n" + chunk

    return combined


def identify(
    text: str,
    model: str = DEFAULT_MODEL,
    ontology_metadata: list[OntologyMetadata] | None = None,
    verbose: bool = False,
) -> str:
    """Run the IDENTIFY agent on paper text.

    For long documents, uses two-pass chunking:
    1. Extract document context (abbreviations, topic, key entities)
    2. Process chunks with context injected

    Args:
        text: Plain text of the paper
        model: LLM model to use
        ontology_metadata: Optional metadata about indexed ontologies
        verbose: Print debug info

    Returns:
        XML-tagged text with concepts
    """
    # Format ontology info if provided
    ontology_info = ""
    if ontology_metadata:
        ontology_info = format_ontology_info(ontology_metadata)

    # Check if we need chunking
    needs_chunking = len(text) > MAX_CHUNK_CHARS

    if not needs_chunking:
        # Simple case: process entire document
        if verbose:
            print(f"Calling LLM ({model}) for identification...")

        result = identify_chunk(
            text, model, ontology_info, document_context="", verbose=verbose
        )

        if verbose:
            print(f"LLM call complete. Response length: {len(result) if result else 0} chars")

        return result

    # Long document: chunk and process
    if verbose:
        print(f"Document too long ({len(text)} chars), using chunked processing...", flush=True)

    chunks = chunk_text(text)

    if verbose:
        chunk_sizes = [len(c) for c in chunks]
        print(f"  Chunks: {len(chunks)} (sizes: {min(chunk_sizes)}-{max(chunk_sizes)}, avg {sum(chunk_sizes)//len(chunks)})", flush=True)
        print(flush=True)

    tagged_chunks = []
    total_concepts = 0
    for i, chunk in enumerate(chunks, 1):
        tagged = identify_chunk(
            chunk, model, ontology_info, document_context="",
            verbose=verbose, chunk_num=i, total_chunks=len(chunks)
        )
        tagged_chunks.append(tagged)

        chunk_concepts = tagged.count("<concept ")
        total_concepts += chunk_concepts
        if verbose:
            print(f"    Concepts: {chunk_concepts}", flush=True)

    # Combine chunks
    result = combine_chunk_outputs(tagged_chunks)
    final_concepts = result.count("<concept ")

    if verbose:
        print(flush=True)
        print(f"Combined: {final_concepts} concepts, {len(result)} chars", flush=True)

    return result


def parse_xml_output(xml_text: str) -> list[Concept]:
    """Parse XML output from IDENTIFY into Concept objects.

    Args:
        xml_text: XML-tagged text from identify()

    Returns:
        List of Concept objects
    """
    concepts = []

    root = parse_xml_fragment(xml_text)
    original_text = strip_tags(xml_text)

    for concept_elem in root.findall(".//concept"):
        concept_text = concept_elem.text or ""
        concept_context = concept_elem.get("context", "")

        # Find position in original text
        start, end = _find_position(original_text, concept_text, concepts)

        if concept_text.strip():
            concepts.append(
                Concept(
                    text=concept_text.strip(),
                    context=concept_context,
                    start=start,
                    end=end,
                )
            )

    return concepts


def _find_position(
    text: str, concept_text: str, existing_concepts: list[Concept]
) -> tuple[int, int]:
    """Find the position of a concept in the original text.

    Tries to find the next occurrence after any existing concepts
    to handle repeated mentions.
    """
    # Start searching after the last found concept
    search_start = 0
    if existing_concepts:
        search_start = existing_concepts[-1].end

    # Find the concept text
    pos = text.find(concept_text, search_start)
    if pos == -1:
        # Try from beginning if not found after last concept
        pos = text.find(concept_text)
    if pos == -1:
        # Not found - return -1, -1
        return -1, -1

    return pos, pos + len(concept_text)
