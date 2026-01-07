"""RELATE agent - extract relationships between disambiguated concepts."""

import asyncio
import json
import re
import time
from pathlib import Path

from ontoagain.index import search_index_batch
from ontoagain.llm import call_llm, call_llm_async, get_client, run_async
from ontoagain.models import OntologyMetadata, RawRelationship, Relationship
from ontoagain.xml_utils import parse_xml_fragment, _escape_attr

# Load prompt templates
EXTRACT_PROMPT_PATH = Path(__file__).parent / "prompts" / "relate_extract.txt"
DISAMBIGUATE_PROMPT_PATH = Path(__file__).parent / "prompts" / "relate_disambiguate.txt"

# Batching parameters for disambiguation
MAX_BATCH_SIZE = 10  # Max relationships per LLM call


def load_extract_prompt() -> str:
    """Load the relationship extraction prompt template."""
    return EXTRACT_PROMPT_PATH.read_text()


def load_disambiguate_prompt() -> str:
    """Load the relationship disambiguation prompt template."""
    return DISAMBIGUATE_PROMPT_PATH.read_text()


def extract_matched_concepts_from_xml(xml_text: str) -> list[dict]:
    """Extract concepts with their ontology matches from disambiguated XML.

    Supports both compact <C n="..." q="..."><M .../></C> and legacy formats.

    Args:
        xml_text: XML with concept tags containing match elements

    Returns:
        List of dicts with keys: text, context, n (concept ID), matches (list of {ontology, id, label})
    """
    root = parse_xml_fragment(xml_text)
    concepts = []

    # Support both <C> (compact) and <concept> (legacy) formats
    for elem in root.findall(".//C") + root.findall(".//concept"):
        matches = []
        # Support both <M> (compact) and <match> (legacy) formats
        for match_elem in elem.findall("M") + elem.findall("match"):
            matches.append({
                "ontology": match_elem.get("o", "") or match_elem.get("ontology", ""),
                "id": match_elem.get("id", ""),
                "label": match_elem.get("l", "") or match_elem.get("label", ""),
            })

        # Get concept text (first text node, before any match elements)
        concept_text = elem.text.strip() if elem.text else ""

        concepts.append({
            "n": elem.get("n", ""),  # Concept ID
            "text": concept_text,
            "context": elem.get("q", "") or elem.get("context", ""),
            "matches": matches,
        })

    return concepts


def format_candidates(candidates: list[dict]) -> str:
    """Format candidate relationship terms for the prompt."""
    lines = []
    for c in candidates:
        synonyms = ", ".join(c.get("synonyms", [])[:5])
        line = f"- {c['id']} ({c['ontology']}): {c['label']}"
        if synonyms:
            line += f" [synonyms: {synonyms}]"
        if c.get("definition"):
            defn = c["definition"][:200]
            if len(c["definition"]) > 200:
                defn += "..."
            line += f"\n  Definition: {defn}"
        lines.append(line)
    return "\n".join(lines)


def format_ontology_context(metadata: list[OntologyMetadata]) -> str:
    """Format ontology metadata as context for disambiguation."""
    if not metadata:
        return ""

    lines = ["## Ontology Context", ""]
    lines.append("Use this to understand what each ontology contains:")
    lines.append("")

    for m in metadata:
        lines.append(f"- **{m.name}**: {m.description}")

    lines.append("")
    return "\n".join(lines)


def relate_extract(
    xml_input: str,
    model: str = "anthropic/claude-sonnet",
    verbose: bool = False,
    max_concurrent: int = 4,
    relationship_metadata: list[OntologyMetadata] | None = None,
) -> list[RawRelationship]:
    """Extract relationships from disambiguated XML.

    Args:
        xml_input: XML with <C n="..."><M id="..."/></C> tags
        model: LLM model to use
        verbose: Print debug info
        max_concurrent: Maximum concurrent LLM calls (for chunking)
        relationship_metadata: Optional metadata from relationship ontology index

    Returns:
        List of RawRelationship objects
    """
    # Extract concepts with their matches
    concepts = extract_matched_concepts_from_xml(xml_input)

    if not concepts:
        return []

    matched_count = sum(1 for c in concepts if c["matches"])
    if verbose:
        print(f"Found {len(concepts)} concepts ({matched_count} with matches)")

    if matched_count == 0:
        if verbose:
            print("  No matched concepts, skipping relation extraction")
        return []

    # Build relationship context from metadata
    relationship_context = ""
    if relationship_metadata:
        lines = ["\n## Relationship Types to Extract\n"]
        for m in relationship_metadata:
            lines.append(f"**{m.name}**: {m.description}")
            if m.term_format:
                lines.append(f"Examples: {m.term_format}")
        lines.append("")
        relationship_context = "\n".join(lines)

    # Build prompt
    prompt = load_extract_prompt()
    prompt = prompt.replace("{relationship_context}", relationship_context)
    prompt = prompt.replace("{xml_input}", xml_input)

    # Call LLM
    client, model_name = get_client(model)

    if verbose:
        print(f"Calling LLM to extract relationships...")

    start = time.time()
    messages = [{"role": "user", "content": prompt}]
    result = call_llm(client, model_name, messages)
    elapsed = time.time() - start

    if verbose:
        print(f"  -> {len(result)} chars in {elapsed:.1f}s")

    # Parse XML result
    relationships = _parse_relation_xml(result, verbose)

    if verbose:
        print(f"  Extracted {len(relationships)} relationships")

    return relationships


def _parse_relation_xml(xml_text: str, verbose: bool = False) -> list[RawRelationship]:
    """Parse relationship XML output from LLM.

    Expected format:
    <relations>
    <R s="ID1" o="ID2" p="predicate" e="1,2,3">Summary text</R>
    </relations>
    """
    relationships = []

    try:
        root = parse_xml_fragment(xml_text)

        for elem in root.findall(".//R"):
            subject_id = elem.get("s", "")
            object_id = elem.get("o", "")
            predicate = elem.get("p", "")
            evidence_ids = elem.get("e", "")
            summary = elem.text.strip() if elem.text else ""

            # Parse concept IDs from comma-separated string
            concept_ids = [x.strip() for x in evidence_ids.split(",") if x.strip()]

            if subject_id and object_id and predicate:
                relationships.append(RawRelationship(
                    subject_id=subject_id,
                    object_id=object_id,
                    predicate=predicate,
                    concept_ids=concept_ids,
                    summary=summary,
                ))
    except Exception as e:
        if verbose:
            print(f"  Warning: Failed to parse relationship XML: {e}")

    return relationships


async def relate_extract_async(
    xml_input: str,
    model: str,
) -> list[RawRelationship]:
    """Async version of relate_extract."""
    # Extract concepts with their matches
    concepts = extract_matched_concepts_from_xml(xml_input)

    if not concepts:
        return []

    matched_count = sum(1 for c in concepts if c["matches"])
    if matched_count == 0:
        return []

    # Build prompt
    prompt = load_extract_prompt()
    prompt = prompt.replace("{xml_input}", xml_input)

    # Call LLM
    messages = [{"role": "user", "content": prompt}]
    result = await call_llm_async(model, messages)

    # Parse XML result
    return _parse_relation_xml(result or "")


def relate_disambiguate(
    relationships: list[RawRelationship],
    relationship_index: Path,
    model: str = "anthropic/claude-sonnet",
    concept_index: Path | None = None,
    top_k: int = 10,
    verbose: bool = False,
    max_concurrent: int = 6,
    ontology_metadata: list[OntologyMetadata] | None = None,
) -> list[Relationship]:
    """Disambiguate relationship predicates to ontology terms.

    Args:
        relationships: List of RawRelationship from relate_extract
        relationship_index: Path to relationship ontology index (e.g., RO)
        model: LLM model to use
        concept_index: Optional path to concept index for subject/object context
        top_k: Number of candidates per predicate
        verbose: Print debug info
        max_concurrent: Maximum concurrent LLM calls
        ontology_metadata: Optional metadata about ontologies

    Returns:
        List of Relationship objects with predicate_id and predicate_label
    """
    if not relationships:
        return []

    if verbose:
        print(f"Disambiguating {len(relationships)} relationship predicates...")

    # Step 1: Get unique predicates and their candidates
    unique_predicates = list(set(r.predicate for r in relationships))

    if verbose:
        print(f"  {len(unique_predicates)} unique predicates")

    # Batch search for candidates
    candidates_by_predicate = {}
    all_candidates = search_index_batch(
        relationship_index, unique_predicates, top_k=top_k, verbose=verbose
    )

    for pred, cands in zip(unique_predicates, all_candidates):
        candidates_by_predicate[pred] = cands

    # Step 2: Batch disambiguate predicates
    # Group relationships into batches
    batches = []
    for i in range(0, len(relationships), MAX_BATCH_SIZE):
        batches.append(relationships[i:i + MAX_BATCH_SIZE])

    if verbose:
        print(f"  Processing {len(batches)} batches...")

    # Process batches
    all_results = run_async(
        _disambiguate_parallel(
            batches, candidates_by_predicate, model,
            max_concurrent, verbose, ontology_metadata
        )
    )

    # Flatten results
    result_relationships = []
    for batch_results in all_results:
        result_relationships.extend(batch_results)

    if verbose:
        matched = sum(1 for r in result_relationships if r.predicate_id)
        print(f"  Disambiguated: {matched}/{len(result_relationships)} predicates matched")

    return result_relationships


async def _disambiguate_batch_async(
    batch: list[RawRelationship],
    candidates_by_predicate: dict[str, list[dict]],
    model: str,
    batch_num: int,
    total_batches: int,
    ontology_metadata: list[OntologyMetadata] | None = None,
) -> list[Relationship]:
    """Async batch disambiguation."""
    # Collect unique candidates for this batch
    candidate_map: dict[str, dict] = {}
    for rel in batch:
        for c in candidates_by_predicate.get(rel.predicate, []):
            candidate_map[c["id"]] = c

    if not candidate_map:
        # No candidates - return with empty predicate_id
        return [
            Relationship(
                subject_id=r.subject_id,
                object_id=r.object_id,
                predicate_id="",
                predicate_label="",
                predicate_raw=r.predicate,
                concept_ids=r.concept_ids,
                summary=r.summary,
            )
            for r in batch
        ]

    # Format relationships for prompt
    rel_lines = []
    for i, rel in enumerate(batch):
        rel_lines.append(
            f'{i}. "{rel.predicate}" (subject: {rel.subject_id}, object: {rel.object_id})'
        )
    relationships_text = "\n".join(rel_lines)

    # Format candidates
    candidates_text = format_candidates(list(candidate_map.values()))

    # Format ontology context
    ontology_context = ""
    if ontology_metadata:
        ontology_context = format_ontology_context(ontology_metadata)

    # Build prompt
    prompt = load_disambiguate_prompt()
    prompt = prompt.replace("{ontology_context}", ontology_context)
    prompt = prompt.replace("{relationships}", relationships_text)
    prompt = prompt.replace("{candidates}", candidates_text)

    # Call LLM
    messages = [{"role": "user", "content": prompt}]
    result = await call_llm_async(model, messages)

    # Parse result
    selections = {}
    try:
        if result:
            match = re.search(r"\{.*\}", result, re.DOTALL)
            if match:
                selections = json.loads(match.group())
    except json.JSONDecodeError:
        pass

    # Build results
    results = []
    for i, rel in enumerate(batch):
        selected_id = selections.get(str(i), "")

        predicate_id = ""
        predicate_label = ""

        if selected_id and selected_id in candidate_map:
            c = candidate_map[selected_id]
            predicate_id = c["id"]
            predicate_label = c["label"]

        results.append(Relationship(
            subject_id=rel.subject_id,
            object_id=rel.object_id,
            predicate_id=predicate_id,
            predicate_label=predicate_label,
            predicate_raw=rel.predicate,
            concept_ids=rel.concept_ids,
            summary=rel.summary,
        ))

    return results


def raw_relationships_to_xml(relationships: list[RawRelationship]) -> str:
    """Convert raw relationships to XML format.

    Output format:
    <relations>
    <R s="ID1" o="ID2" p="predicate" e="1,2">Summary text</R>
    </relations>
    """
    lines = ["<relations>"]
    for rel in relationships:
        e_attr = ",".join(rel.concept_ids) if rel.concept_ids else ""
        summary = _escape_attr(rel.summary) if rel.summary else ""
        lines.append(
            f'<R s="{rel.subject_id}" o="{rel.object_id}" p="{rel.predicate}" e="{e_attr}">{summary}</R>'
        )
    lines.append("</relations>")
    return "\n".join(lines)


def relationships_to_xml(relationships: list[Relationship]) -> str:
    """Convert disambiguated relationships to XML format.

    Output format:
    <relations>
    <R s="ID1" o="ID2" p="RO:001" pl="causes" pr="induces" e="1,2">Summary text</R>
    </relations>

    Attributes:
    - s = subject ontology ID
    - o = object ontology ID
    - p = predicate ontology ID (from disambiguation)
    - pl = predicate label
    - pr = predicate raw (original from extraction)
    - e = evidence concept IDs
    """
    lines = ["<relations>"]
    for rel in relationships:
        e_attr = ",".join(rel.concept_ids) if rel.concept_ids else ""
        summary = _escape_attr(rel.summary) if rel.summary else ""
        p_attr = rel.predicate_id or ""
        pl_attr = _escape_attr(rel.predicate_label) if rel.predicate_label else ""
        pr_attr = _escape_attr(rel.predicate_raw) if rel.predicate_raw else ""
        lines.append(
            f'<R s="{rel.subject_id}" o="{rel.object_id}" p="{p_attr}" pl="{pl_attr}" pr="{pr_attr}" e="{e_attr}">{summary}</R>'
        )
    lines.append("</relations>")
    return "\n".join(lines)


async def _disambiguate_parallel(
    batches: list[list[RawRelationship]],
    candidates_by_predicate: dict[str, list[dict]],
    model: str,
    max_concurrent: int,
    verbose: bool,
    ontology_metadata: list[OntologyMetadata] | None,
) -> list[list[Relationship]]:
    """Process batches in parallel."""
    semaphore = asyncio.Semaphore(max_concurrent)
    total_batches = len(batches)

    async def process_with_limit(batch: list[RawRelationship], batch_num: int):
        async with semaphore:
            if verbose:
                print(f"  [{batch_num}/{total_batches}] {len(batch)} relationships...", flush=True)

            result = await _disambiguate_batch_async(
                batch, candidates_by_predicate, model,
                batch_num, total_batches, ontology_metadata
            )

            if verbose:
                matched = sum(1 for r in result if r.predicate_id)
                print(f"  [{batch_num}/{total_batches}] Done: {matched}/{len(batch)} matched", flush=True)

            return result

    tasks = [process_with_limit(batch, i + 1) for i, batch in enumerate(batches)]
    return await asyncio.gather(*tasks)
