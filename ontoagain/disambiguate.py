"""DISAMBIGUATE agent - map concepts to ontology terms."""

import json
import re
import time
from pathlib import Path

from ontoagain.index import search_index, search_index_batch
from ontoagain.llm import call_llm, get_client
from ontoagain.models import Concept, OntologyMatch, OntologyMetadata, TaggedConcept
from ontoagain.xml_utils import extract_concepts_from_xml, update_xml_with_matches

# Load prompt templates
PROMPT_PATH = Path(__file__).parent / "prompts" / "disambiguate.txt"
BATCH_PROMPT_PATH = Path(__file__).parent / "prompts" / "disambiguate_batch.txt"

# Batching parameters
MIN_OVERLAP_RATIO = 0.3  # 30% candidate overlap to join a cluster
MIN_BATCH_SIZE = 5  # Minimum concepts per LLM call (merge small clusters)
MAX_BATCH_SIZE = 20  # Max concepts per LLM call


def load_prompt() -> str:
    """Load the single-concept disambiguate prompt template."""
    return PROMPT_PATH.read_text()


def load_batch_prompt() -> str:
    """Load the batch disambiguate prompt template."""
    return BATCH_PROMPT_PATH.read_text()


def format_candidates(candidates: list[dict]) -> str:
    """Format candidate terms for the prompt."""
    lines = []
    for c in candidates:
        synonyms = ", ".join(c.get("synonyms", [])[:3])  # Limit synonyms shown
        line = f"- {c['id']} ({c['ontology']}): {c['label']}"
        if synonyms:
            line += f" [synonyms: {synonyms}]"
        if c.get("definition"):
            # Truncate long definitions
            defn = c["definition"][:200]
            if len(c["definition"]) > 200:
                defn += "..."
            line += f"\n  Definition: {defn}"
        lines.append(line)
    return "\n".join(lines)


def format_ontology_context(metadata: list[OntologyMetadata]) -> str:
    """Format ontology metadata as context for disambiguation.

    This helps the LLM understand what each ontology is for and
    reject inappropriate matches (e.g., NCBITaxon for non-species).

    Args:
        metadata: List of ontology metadata

    Returns:
        Formatted string with ontology context
    """
    if not metadata:
        return ""

    lines = ["## Ontology Context", ""]
    lines.append("Use this to understand what each ontology contains:")
    lines.append("")

    for m in metadata:
        lines.append(f"- **{m.name}**: {m.description}")

    lines.append("")
    return "\n".join(lines)


def compute_overlap(set_a: set, set_b: set) -> float:
    """Compute overlap ratio between two sets (Jaccard-like, based on smaller set)."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    smaller = min(len(set_a), len(set_b))
    return intersection / smaller


def cluster_by_candidates(
    concept_candidates: list[tuple[int, set[str]]]
) -> list[list[int]]:
    """Cluster concept indices by candidate overlap.

    Uses greedy clustering: each concept joins the first cluster with sufficient overlap,
    or starts a new cluster. Then merges small clusters to meet MIN_BATCH_SIZE.

    Args:
        concept_candidates: List of (concept_index, set of candidate IDs)

    Returns:
        List of clusters, each cluster is a list of concept indices
    """
    clusters: list[tuple[set[str], list[int]]] = []  # (union of candidates, indices)

    # First pass: cluster by overlap
    for idx, candidates in concept_candidates:
        best_cluster = None
        best_overlap = 0.0

        for i, (cluster_cands, _) in enumerate(clusters):
            if len(clusters[i][1]) >= MAX_BATCH_SIZE:
                continue
            overlap = compute_overlap(candidates, cluster_cands)
            if overlap >= MIN_OVERLAP_RATIO and overlap > best_overlap:
                best_overlap = overlap
                best_cluster = i

        if best_cluster is not None:
            # Join existing cluster
            clusters[best_cluster][0].update(candidates)
            clusters[best_cluster][1].append(idx)
        else:
            # Start new cluster
            clusters.append((candidates.copy(), [idx]))

    # Second pass: merge small clusters to meet MIN_BATCH_SIZE
    merged: list[list[int]] = []
    pending: list[int] = []  # Accumulate small cluster indices

    for _, indices in clusters:
        if len(indices) >= MIN_BATCH_SIZE:
            # Large enough, keep as-is
            merged.append(indices)
        else:
            # Small cluster, accumulate
            pending.extend(indices)
            if len(pending) >= MIN_BATCH_SIZE:
                # Flush when we have enough
                merged.append(pending[:MAX_BATCH_SIZE])
                pending = pending[MAX_BATCH_SIZE:]

    # Handle remaining pending items
    if pending:
        if merged and len(merged[-1]) + len(pending) <= MAX_BATCH_SIZE:
            # Add to last batch if it fits
            merged[-1].extend(pending)
        else:
            # Create final batch (even if small)
            merged.append(pending)

    return merged


def disambiguate_batch(
    concepts: list[Concept],
    all_candidates: list[list[dict]],
    indices: list[int],
    model: str,
    verbose: bool = False,
    batch_num: int = 0,
    total_batches: int = 0,
    ontology_metadata: list[OntologyMetadata] | None = None,
) -> dict[int, list[OntologyMatch]]:
    """Disambiguate a batch of concepts in one LLM call.

    Args:
        concepts: Full list of concepts
        all_candidates: Candidates for each concept (parallel to concepts)
        indices: Which concept indices to process in this batch
        model: LLM model to use
        verbose: Print debug info
        batch_num: Current batch number for progress
        total_batches: Total number of batches for progress
        ontology_metadata: Optional metadata about ontologies

    Returns:
        Dict mapping concept index to list of OntologyMatch
    """
    # Collect unique candidates across the batch
    candidate_map: dict[str, dict] = {}
    for idx in indices:
        for c in all_candidates[idx]:
            candidate_map[c["id"]] = c

    if not candidate_map:
        return {idx: [] for idx in indices}

    # Format concepts for prompt
    concept_lines = []
    for i, idx in enumerate(indices):
        c = concepts[idx]
        concept_lines.append(f"{i}. **{c.text}**: {c.context}")
    concepts_text = "\n".join(concept_lines)

    # Format candidates
    candidates_text = format_candidates(list(candidate_map.values()))

    # Format ontology context
    ontology_context = ""
    if ontology_metadata:
        ontology_context = format_ontology_context(ontology_metadata)

    # Build prompt
    prompt = load_batch_prompt()
    prompt = prompt.replace("{ontology_context}", ontology_context)
    prompt = prompt.replace("{concepts}", concepts_text)
    prompt = prompt.replace("{candidates}", candidates_text)

    # Call LLM
    client, model_name = get_client(model)

    if verbose:
        concept_names = [concepts[idx].text for idx in indices]
        first_concept = concept_names[0][:40] + ("..." if len(concept_names[0]) > 40 else "")
        batch_info = f"[{batch_num}/{total_batches}]" if total_batches else ""
        print(f"  {batch_info} {len(indices)} concepts: {first_concept} ...", flush=True)

    start = time.time()
    messages = [{"role": "user", "content": prompt}]
    result = call_llm(client, model_name, messages)
    elapsed = time.time() - start

    if verbose:
        print(f"    -> {len(result)} chars in {elapsed:.1f}s", flush=True)

    # Parse result
    try:
        # Try to extract JSON object from response
        match = re.search(r"\{.*\}", result, re.DOTALL)
        if match:
            selections = json.loads(match.group())
        else:
            selections = {}
    except json.JSONDecodeError:
        selections = {}

    # Build matches for each concept
    results: dict[int, list[OntologyMatch]] = {}
    for i, idx in enumerate(indices):
        selected_ids = selections.get(str(i), [])
        matches = []
        for term_id in selected_ids:
            if term_id in candidate_map:
                c = candidate_map[term_id]
                matches.append(
                    OntologyMatch(
                        ontology=c["ontology"],
                        id=c["id"],
                        label=c["label"],
                    )
                )
        results[idx] = matches

    return results


def disambiguate_concept(
    concept: Concept,
    index_path: Path,
    model: str = "claude-sonnet-4-20250514",
    top_k: int = 20,
    verbose: bool = False,
) -> TaggedConcept:
    """Disambiguate a single concept to ontology terms.

    Args:
        concept: Concept from IDENTIFY
        index_path: Path to LanceDB index
        model: LLM model to use
        top_k: Number of candidates to retrieve
        verbose: Print debug info

    Returns:
        TaggedConcept with ontology matches
    """
    # Retrieve candidates
    query = concept.context if concept.context else concept.text
    candidates = search_index(index_path, query, top_k=top_k)

    if not candidates:
        return TaggedConcept(
            text=concept.text,
            context=concept.context,
            start=concept.start,
            end=concept.end,
            matches=[],
        )

    # Build prompt
    prompt_template = load_prompt()
    prompt = prompt_template.replace("{concept_text}", concept.text)
    prompt = prompt.replace("{concept_context}", concept.context)
    prompt = prompt.replace("{candidates}", format_candidates(candidates))

    # Call LLM
    client, model_name = get_client(model)

    if verbose:
        print(f"  Disambiguating: {concept.text}")

    messages = [{"role": "user", "content": prompt}]
    result = call_llm(client, model_name, messages)

    # Parse result
    try:
        selected_ids = json.loads(result.strip())
    except json.JSONDecodeError:
        # Try to extract JSON array from response
        import re
        match = re.search(r"\[.*?\]", result, re.DOTALL)
        if match:
            selected_ids = json.loads(match.group())
        else:
            selected_ids = []

    # Build matches from selected IDs
    matches = []
    candidate_map = {c["id"]: c for c in candidates}
    for term_id in selected_ids:
        if term_id in candidate_map:
            c = candidate_map[term_id]
            matches.append(
                OntologyMatch(
                    ontology=c["ontology"],
                    id=c["id"],
                    label=c["label"],
                )
            )

    return TaggedConcept(
        text=concept.text,
        context=concept.context,
        start=concept.start,
        end=concept.end,
        matches=matches,
    )


def disambiguate(
    xml_input: str,
    index_path: Path,
    model: str = "claude-sonnet-4-20250514",
    top_k: int = 20,
    verbose: bool = False,
    ontology_metadata: list[OntologyMetadata] | None = None,
) -> tuple[str, dict]:
    """Disambiguate concepts in XML to ontology terms.

    Uses batching to reduce LLM calls: concepts with overlapping candidate sets
    are grouped and processed together.

    Args:
        xml_input: XML text with <concept> tags from IDENTIFY
        index_path: Path to LanceDB index
        model: LLM model to use
        top_k: Number of candidates per concept
        verbose: Print debug info
        ontology_metadata: Optional metadata about ontologies for context

    Returns:
        Tuple of (updated_xml, stats)
    """
    # Extract concepts from XML
    concept_dicts = extract_concepts_from_xml(xml_input)

    if not concept_dicts:
        return xml_input, {"total": 0, "matched": 0, "unmatched": 0, "total_mappings": 0}

    if verbose:
        print(f"Disambiguating {len(concept_dicts)} concepts...")

    # Step 1: Retrieve candidates for all concepts (batch search - loads model once)
    if verbose:
        print("  Retrieving candidates from vector index...")

    # Use context for vector search (replace semicolons with spaces for embedding)
    queries = []
    for c in concept_dicts:
        if c["context"]:
            queries.append(c["context"].replace(";", " "))
        else:
            queries.append(c["text"])
    all_candidates = search_index_batch(index_path, queries, top_k=top_k, verbose=verbose)

    # Step 2: Cluster concepts by candidate overlap
    concept_candidate_sets = [
        (i, {c["id"] for c in cands})
        for i, cands in enumerate(all_candidates)
    ]
    clusters = cluster_by_candidates(concept_candidate_sets)

    if verbose:
        print(f"  Grouped into {len(clusters)} batches (from {len(concept_dicts)} concepts)")

    # Convert dicts to Concept objects for batch processing
    concepts = [
        Concept(
            text=c["text"],
            context=c["context"],
            start=0,  # Not used in XML flow
            end=0,
        )
        for c in concept_dicts
    ]

    # Step 3: Disambiguate each cluster
    all_matches: dict[int, list[OntologyMatch]] = {}
    total_batches = len(clusters)
    for batch_num, cluster in enumerate(clusters, 1):
        batch_results = disambiguate_batch(
            concepts, all_candidates, cluster, model, verbose,
            batch_num=batch_num, total_batches=total_batches,
            ontology_metadata=ontology_metadata,
        )
        all_matches.update(batch_results)

    # Step 4: Build matches list parallel to concepts for XML update
    matches_list: list[list[dict]] = []
    matched_count = 0
    total_mappings = 0

    for i in range(len(concept_dicts)):
        matches = all_matches.get(i, [])
        # Convert OntologyMatch to dict for XML update
        match_dicts = [
            {"ontology": m.ontology, "id": m.id, "label": m.label}
            for m in matches
        ]
        matches_list.append(match_dicts)

        if matches:
            matched_count += 1
            total_mappings += len(matches)

    # Step 5: Update XML with matches
    updated_xml = update_xml_with_matches(xml_input, matches_list)

    stats = {
        "total": len(concept_dicts),
        "batches": len(clusters),
        "matched": matched_count,
        "unmatched": len(concept_dicts) - matched_count,
        "total_mappings": total_mappings,
    }

    if verbose:
        print(f"  Matched: {matched_count}, Unmatched: {stats['unmatched']}")

    return updated_xml, stats
