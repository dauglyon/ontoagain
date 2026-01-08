"""DISAMBIGUATE agent - map concepts to ontology terms (batch processing)."""

import asyncio
import json
import re
import time
from pathlib import Path

from ontoagain.index import search_index_batch
from ontoagain.llm import call_llm, call_llm_async, get_client, run_async
from ontoagain.models import Concept, OntologyMatch, OntologyMetadata
from ontoagain.xml_utils import extract_concepts_from_xml, update_xml_with_matches

# Load prompt template
BATCH_PROMPT_PATH = Path(__file__).parent / "prompts" / "disambiguate_batch.txt"

# Batching parameters
MIN_OVERLAP_RATIO = 0.3  # 30% candidate overlap to join a cluster
MIN_BATCH_SIZE = 5  # Minimum concepts per LLM call (merge small clusters)
MAX_BATCH_SIZE = 20  # Max concepts per LLM call


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


async def disambiguate_batch_async(
    concepts: list[Concept],
    all_candidates: list[list[dict]],
    indices: list[int],
    model: str,
    batch_num: int,
    total_batches: int,
    ontology_metadata: list[OntologyMetadata] | None = None,
) -> tuple[int, dict[int, list[OntologyMatch]]]:
    """Async version of disambiguate_batch for parallel processing.

    Args:
        concepts: Full list of concepts
        all_candidates: Candidates for each concept (parallel to concepts)
        indices: Which concept indices to process in this batch
        model: LLM model to use
        batch_num: Current batch number for progress
        total_batches: Total number of batches for progress
        ontology_metadata: Optional metadata about ontologies

    Returns:
        Tuple of (batch_num, dict mapping concept index to list of OntologyMatch)
    """
    # Collect unique candidates across the batch
    candidate_map: dict[str, dict] = {}
    for idx in indices:
        for c in all_candidates[idx]:
            candidate_map[c["id"]] = c

    if not candidate_map:
        return batch_num, {idx: [] for idx in indices}

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

    # Call LLM async
    messages = [{"role": "user", "content": prompt}]
    result = await call_llm_async(model, messages)

    # Parse result
    try:
        if result is None:
            selections = {}
        else:
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

    return batch_num, results


async def disambiguate_parallel(
    clusters: list[list[int]],
    concepts: list[Concept],
    all_candidates: list[list[dict]],
    model: str,
    max_concurrent: int = 6,
    verbose: bool = False,
    ontology_metadata: list[OntologyMetadata] | None = None,
) -> dict[int, list[OntologyMatch]]:
    """Process batches in parallel with concurrency limit.

    Args:
        clusters: List of clusters (each cluster is a list of concept indices)
        concepts: Full list of concepts
        all_candidates: Candidates for each concept
        model: LLM model to use
        max_concurrent: Maximum concurrent LLM calls
        verbose: Print progress info
        ontology_metadata: Optional metadata about ontologies

    Returns:
        Dict mapping concept index to list of OntologyMatch
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    total_batches = len(clusters)
    start_time = time.time()

    async def process_with_limit(cluster: list[int], batch_num: int) -> tuple[int, dict[int, list[OntologyMatch]]]:
        async with semaphore:
            batch_start = time.time()
            if verbose:
                first_concept = concepts[cluster[0]].text[:40]
                print(f"  [{batch_num}/{total_batches}] {len(cluster)} concepts: {first_concept}...", flush=True)

            result = await disambiguate_batch_async(
                concepts, all_candidates, cluster, model,
                batch_num, total_batches, ontology_metadata
            )

            elapsed = time.time() - batch_start
            if verbose:
                matched = sum(1 for idx in cluster if result[1].get(idx))
                print(f"  [{batch_num}/{total_batches}] Done: {matched}/{len(cluster)} matched ({elapsed:.1f}s)", flush=True)
            return result

    # Launch all tasks
    tasks = [process_with_limit(cluster, i + 1) for i, cluster in enumerate(clusters)]
    results = await asyncio.gather(*tasks)

    # Merge all results
    all_matches: dict[int, list[OntologyMatch]] = {}
    for _, batch_results in results:
        all_matches.update(batch_results)

    if verbose:
        total_elapsed = time.time() - start_time
        matched_count = sum(1 for m in all_matches.values() if m)
        print(f"\nParallel processing complete: {matched_count}/{len(concepts)} matched in {total_elapsed:.1f}s", flush=True)

    return all_matches


def disambiguate(
    xml_input: str,
    index_path: Path,
    model: str = "claude-sonnet-4-20250514",
    top_k: int = 20,
    verbose: bool = False,
    ontology_metadata: list[OntologyMetadata] | None = None,
    max_concurrent: int = 6,
) -> tuple[list[tuple[str, str]], dict]:
    """Disambiguate concepts in XML to ontology terms.

    Expects <Documents><D id="...">...</D></Documents> format.
    Concepts are pooled across documents for efficient batching.

    Args:
        xml_input: XML with <Documents><D>...</D></Documents> structure
        index_path: Path to LanceDB index
        model: LLM model to use
        top_k: Number of candidates per concept
        verbose: Print debug info
        ontology_metadata: Optional metadata about ontologies for context
        max_concurrent: Maximum concurrent LLM calls

    Returns:
        Tuple of (list of (doc_id, xml_content) tuples, stats dict)
    """
    from ontoagain.xml_utils import extract_documents

    documents = extract_documents(xml_input)

    # Extract concepts from all documents, tracking source
    all_concept_dicts = []
    doc_ranges = {}  # doc_id -> (start_idx, end_idx)

    for doc_id, doc_xml in documents:
        start = len(all_concept_dicts)
        all_concept_dicts.extend(extract_concepts_from_xml(doc_xml))
        doc_ranges[doc_id] = (start, len(all_concept_dicts))

    if not all_concept_dicts:
        return xml_input, {"total": 0, "matched": 0, "unmatched": 0, "total_mappings": 0}

    if verbose:
        print(f"Disambiguating {len(all_concept_dicts)} concepts from {len(documents)} doc(s)...")
        print("  Retrieving candidates from vector index...")

    # Step 1: Retrieve candidates for all concepts
    queries = [
        c["context"].replace(";", " ") if c["context"] else c["text"]
        for c in all_concept_dicts
    ]
    all_candidates = search_index_batch(index_path, queries, top_k=top_k, verbose=verbose)

    # Step 2: Cluster concepts by candidate overlap (across all documents)
    clusters = cluster_by_candidates([
        (i, {c["id"] for c in cands})
        for i, cands in enumerate(all_candidates)
    ])

    if verbose:
        print(f"  Grouped into {len(clusters)} batches")

    # Step 3: Disambiguate
    concepts = [
        Concept(text=c["text"], context=c["context"], start=0, end=0)
        for c in all_concept_dicts
    ]
    all_matches = run_async(
        disambiguate_parallel(
            clusters, concepts, all_candidates, model,
            max_concurrent, verbose, ontology_metadata
        )
    )

    # Step 4: Build matches list
    matches_list = []
    matched_count = 0
    total_mappings = 0

    for i in range(len(all_concept_dicts)):
        matches = all_matches.get(i, [])
        match_dicts = [{"ontology": m.ontology, "id": m.id, "label": m.label} for m in matches]
        matches_list.append(match_dicts)
        if matches:
            matched_count += 1
            total_mappings += len(matches)

    # Step 5: Update each document's XML
    updated_docs = []
    for doc_id, doc_xml in documents:
        start, end = doc_ranges[doc_id]
        updated_doc = update_xml_with_matches(doc_xml, matches_list[start:end])
        updated_docs.append((doc_id, updated_doc))

    stats = {
        "total": len(all_concept_dicts),
        "batches": len(clusters),
        "matched": matched_count,
        "unmatched": len(all_concept_dicts) - matched_count,
        "total_mappings": total_mappings,
        "documents": len(documents),
    }

    if verbose:
        print(f"  Matched: {matched_count}, Unmatched: {stats['unmatched']}")

    return updated_docs, stats
