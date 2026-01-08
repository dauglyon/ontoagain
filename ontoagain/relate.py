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

    Expects <Documents><D id="...">...</D></Documents> format.
    Processes documents in parallel.

    Args:
        xml_input: XML with Documents/D structure containing <C><M/></C> tags
        model: LLM model to use
        verbose: Print debug info
        max_concurrent: Maximum concurrent LLM calls
        relationship_metadata: Optional metadata from relationship ontology index

    Returns:
        List of RawRelationship objects (each has doc_id field)
    """
    from ontoagain.xml_utils import extract_documents

    documents = extract_documents(xml_input)

    if verbose:
        print(f"Extracting relationships from {len(documents)} doc(s)...")

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

    async def process_all():
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_doc(doc_id: str, doc_xml: str) -> list[RawRelationship]:
            concepts = extract_matched_concepts_from_xml(doc_xml)
            matched = sum(1 for c in concepts if c["matches"])
            if matched == 0:
                return []

            async with semaphore:
                prompt = load_extract_prompt()
                prompt = prompt.replace("{relationship_context}", relationship_context)
                prompt = prompt.replace("{xml_input}", doc_xml)

                messages = [{"role": "user", "content": prompt}]
                result = await call_llm_async(model, messages)
                rels = _parse_relation_xml(doc_id, result or "")

                if verbose:
                    print(f"  {doc_id}: {len(rels)} relationships", flush=True)

                return rels

        tasks = [process_doc(doc_id, doc_xml) for doc_id, doc_xml in documents]
        return await asyncio.gather(*tasks)

    results = run_async(process_all())

    # Flatten all results
    all_rels = []
    for doc_rels in results:
        all_rels.extend(doc_rels)

    return all_rels


def _parse_relation_xml(doc_id: str, xml_text: str, verbose: bool = False) -> list[RawRelationship]:
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
                    doc_id=doc_id,
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
        relationships: List of RawRelationship from relate_extract (each has doc_id)
        relationship_index: Path to relationship ontology index (e.g., RO)
        model: LLM model to use
        concept_index: Optional path to concept index for subject/object context
        top_k: Number of candidates per predicate
        verbose: Print debug info
        max_concurrent: Maximum concurrent LLM calls
        ontology_metadata: Optional metadata about ontologies

    Returns:
        List of Relationship objects (each has doc_id from its RawRelationship)
    """
    if not relationships:
        return []

    if verbose:
        doc_count = len(set(r.doc_id for r in relationships))
        print(f"Disambiguating {len(relationships)} relationships from {doc_count} doc(s)...")

    # Step 1: Get unique predicates and their candidates
    unique_predicates = list(set(r.predicate for r in relationships))

    if verbose:
        print(f"  {len(unique_predicates)} unique predicates")

    candidates_by_predicate = {}
    all_candidates = search_index_batch(
        relationship_index, unique_predicates, top_k=top_k, verbose=verbose
    )
    for pred, cands in zip(unique_predicates, all_candidates):
        candidates_by_predicate[pred] = cands

    # Step 2: Batch disambiguate
    batches = [relationships[i:i + MAX_BATCH_SIZE] for i in range(0, len(relationships), MAX_BATCH_SIZE)]

    if verbose:
        print(f"  Processing {len(batches)} batches...")

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
                doc_id=r.doc_id,
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
            doc_id=rel.doc_id,
            subject_id=rel.subject_id,
            object_id=rel.object_id,
            predicate_id=predicate_id,
            predicate_label=predicate_label,
            predicate_raw=rel.predicate,
            concept_ids=rel.concept_ids,
            summary=rel.summary,
        ))

    return results


def parse_raw_relationships_from_xml(xml_text: str) -> list[RawRelationship]:
    """Parse raw relationships from XML format.

    Expected format:
    <relations>
    <R d="doc_id" s="ID1" o="ID2" p="predicate" e="1,2">Summary text</R>
    </relations>

    Args:
        xml_text: XML string with <relations><R/></relations> structure

    Returns:
        List of RawRelationship objects
    """
    root = parse_xml_fragment(xml_text)
    relationships = []

    for elem in root.findall(".//R"):
        evidence_ids = elem.get("e", "")
        concept_ids = [x.strip() for x in evidence_ids.split(",") if x.strip()]
        relationships.append(RawRelationship(
            doc_id=elem.get("d", ""),
            subject_id=elem.get("s", ""),
            object_id=elem.get("o", ""),
            predicate=elem.get("p", ""),
            concept_ids=concept_ids,
            summary=elem.text.strip() if elem.text else "",
        ))

    return relationships


def raw_relationships_to_xml(relationships: list[RawRelationship]) -> str:
    """Convert raw relationships to XML format.

    Output format:
    <relations>
    <R d="doc_id" s="ID1" o="ID2" p="predicate" e="1,2">Summary text</R>
    </relations>
    """
    lines = ["<relations>"]
    for rel in relationships:
        e_attr = ",".join(rel.concept_ids) if rel.concept_ids else ""
        summary = _escape_attr(rel.summary) if rel.summary else ""
        lines.append(
            f'<R d="{_escape_attr(rel.doc_id)}" s="{rel.subject_id}" o="{rel.object_id}" p="{rel.predicate}" e="{e_attr}">{summary}</R>'
        )
    lines.append("</relations>")
    return "\n".join(lines)


def relationships_to_xml(relationships: list[Relationship]) -> str:
    """Convert disambiguated relationships to XML format.

    Output format:
    <relations>
    <R d="doc_id" s="ID1" o="ID2" p="RO:001" pl="causes" pr="induces" e="1,2">Summary text</R>
    </relations>

    Attributes:
    - d = document ID
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
            f'<R d="{_escape_attr(rel.doc_id)}" s="{rel.subject_id}" o="{rel.object_id}" p="{p_attr}" pl="{pl_attr}" pr="{pr_attr}" e="{e_attr}">{summary}</R>'
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
