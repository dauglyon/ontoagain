"""Ontology indexing using OAK and LanceDB.

Optimized for large ontologies with:
- GPU acceleration when available (FP16)
- Chunked processing for memory efficiency
- Parallel ontology parsing
- Progress tracking
"""

import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lancedb
import torch
from oaklib import get_adapter
from sentence_transformers import SentenceTransformer

from ontoagain.models import OntologiesConfig, OntologyMetadata, OntologyTerm

# Default embedding model
DEFAULT_MODEL = "BAAI/bge-m3"

# Processing defaults - conservative for GPU memory
DEFAULT_BATCH_SIZE_GPU = 32  # Small batch to avoid OOM on 11GB GPU
DEFAULT_BATCH_SIZE_CPU = 64
DEFAULT_CHUNK_SIZE = 10000  # Terms per chunk (smaller for memory)
DEFAULT_WORKERS = 4  # Parallel ontology parsing


def load_mesh_ascii(path: Path) -> list[OntologyTerm]:
    """Load MESH from NLM ASCII format (.bin file).

    Supports both Descriptors (d*.bin) and Supplementary Concepts (c*.bin):
    - Records separated by *NEWRECORD
    - Descriptors: MH (label), UI (id), MS (definition), ENTRY (synonyms)
    - Supplementary: NM (label), UI (id), NO (definition), SY (synonyms)

    Args:
        path: Path to MESH .bin file (e.g., d2025.bin or c2025.bin)

    Returns:
        List of OntologyTerm objects
    """
    terms = []
    current: dict = {"synonyms": []}
    record_count = 0

    def save_current():
        """Save current record if valid."""
        label = current.get("MH") or current.get("NM")
        uid = current.get("UI")
        if uid and label:
            definition = current.get("MS") or current.get("NO") or ""
            terms.append(
                OntologyTerm(
                    id=uid,
                    ontology="MESH",
                    label=label,
                    synonyms=current.get("synonyms", []),
                    definition=definition,
                )
            )

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            if line == "*NEWRECORD":
                save_current()
                current = {"synonyms": []}
                record_count += 1
                if record_count % 50000 == 0:
                    print(f"    Parsed {record_count} records...", flush=True)
                continue

            if " = " not in line:
                continue

            key, _, value = line.partition(" = ")
            key = key.strip()

            # Label: MH (Descriptor) or NM (Supplementary)
            if key in ("MH", "NM"):
                current[key] = value
            # ID
            elif key == "UI":
                current["UI"] = value
            # Definition: MS (Descriptor) or NO (Supplementary)
            elif key in ("MS", "NO"):
                current[key] = value
            # Synonyms: ENTRY/PRINT ENTRY (Descriptor) or SY (Supplementary)
            elif key in ("ENTRY", "PRINT ENTRY", "SY"):
                # Format: "synonym|T109|T195|..." - extract just the synonym
                synonym = value.split("|")[0].strip()
                label = current.get("MH") or current.get("NM")
                if synonym and synonym != label:
                    current["synonyms"].append(synonym)

        # Don't forget the last record
        save_current()

    return terms


def load_obo(path: Path) -> list[OntologyTerm]:
    """Load an OBO/OWL ontology file using OAK (standard format).

    This is the standard loader for OBO Foundry ontologies and OWL files.

    Args:
        path: Path to OBO or OWL file

    Returns:
        List of OntologyTerm objects
    """
    adapter = get_adapter(str(path))
    ontology_name = path.stem.upper()

    terms = []
    for entity in adapter.entities():
        # Skip blank nodes and non-class entities
        if not entity or entity.startswith("_:"):
            continue

        # Get label
        label = adapter.label(entity)
        if not label:
            continue  # Skip terms without labels

        # Check if obsolete
        try:
            if adapter.entity_metadata_map(entity).get("deprecated", False):
                continue
        except Exception:
            pass  # Not all adapters support this

        # Get definition
        definition = adapter.definition(entity) or ""

        # Get synonyms
        try:
            synonyms = list(adapter.entity_aliases(entity))
            # Remove label from synonyms if present
            synonyms = [s for s in synonyms if s != label]
        except Exception:
            synonyms = []

        terms.append(
            OntologyTerm(
                id=entity,
                ontology=ontology_name,
                label=label,
                synonyms=synonyms,
                definition=definition,
            )
        )

    return terms


def load_ontology(path: Path) -> list[OntologyTerm]:
    """Load an ontology file and extract terms.

    Dispatcher that detects format and calls the appropriate loader:
    - MESH ASCII format (.bin files) -> load_mesh_ascii()
    - OBO/OWL files (standard) -> load_obo()

    Args:
        path: Path to ontology file

    Returns:
        List of OntologyTerm objects
    """
    # Special case: MESH ASCII format (.bin files from NLM)
    if path.suffix == ".bin" or "mesh" in path.name.lower():
        try:
            terms = load_mesh_ascii(path)
            if terms:
                return terms
        except Exception:
            pass  # Fall through to standard OBO loader

    # Standard: OBO/OWL via OAK
    return load_obo(path)


def create_embedding_text(term: OntologyTerm) -> str:
    """Create text for embedding from an ontology term.

    Combines label, synonyms, and definition for richer semantic matching.
    """
    parts = [term.label]
    if term.synonyms:
        parts.extend(term.synonyms)
    if term.definition:
        parts.append(term.definition)
    return " | ".join(parts)


def search_index_batch(
    index_path: Path,
    queries: list[str],
    top_k: int = 20,
    embedding_model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> list[list[dict]]:
    """Search the ontology index for multiple queries efficiently.

    Loads the embedding model once for all queries.

    Args:
        index_path: Path to LanceDB database
        queries: List of search queries
        top_k: Number of results per query
        embedding_model: Name of sentence-transformers model
        verbose: Print progress info

    Returns:
        List of results for each query
    """
    if not queries:
        return []

    # Load model (CPU for search to avoid GPU memory issues during inference)
    if verbose:
        print(f"  Loading embedding model...")
    model = SentenceTransformer(embedding_model, device="cpu")

    # Encode all queries at once
    if verbose:
        print(f"  Encoding {len(queries)} queries...", flush=True)
    query_embeddings = model.encode(queries, show_progress_bar=False)
    if verbose:
        print(f"  Encoding complete.", flush=True)

    # Connect to LanceDB
    if verbose:
        print(f"  Searching index for {len(queries)} queries...", flush=True)
    db = lancedb.connect(str(index_path))
    table = db.open_table("terms")

    # Search for each query (brute-force, no ANN index)
    all_results = []
    for i, embedding in enumerate(query_embeddings):
        if verbose and (i + 1) % 100 == 0:
            print(f"    Searched {i + 1}/{len(queries)} queries...", flush=True)

        results = table.search(embedding.tolist()).limit(top_k).to_list()
        # Clean up results (remove vector field)
        for r in results:
            r.pop("vector", None)
        all_results.append(results)

    return all_results


def build_index_from_config(
    config: OntologiesConfig,
    output_path: Path,
    base_path: Path | None = None,
    embedding_model: str = DEFAULT_MODEL,
    batch_size: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = False,
) -> dict:
    """Build a LanceDB index from ontology config.

    This is the preferred method for building indexes as it stores
    ontology metadata (descriptions, term formats) alongside the terms.

    Args:
        config: Ontologies configuration with paths and descriptions
        output_path: Directory for LanceDB database
        base_path: Base path for resolving relative ontology paths
        embedding_model: Name of sentence-transformers model
        batch_size: Batch size for encoding (auto-detected if None)
        chunk_size: Number of terms to process per chunk
        workers: Number of parallel workers for ontology parsing
        verbose: Print progress info

    Returns:
        Stats dict with counts per ontology
    """
    # Resolve paths
    if base_path is None:
        base_path = Path.cwd()

    ontology_configs = []
    for onto_config in config.ontologies:
        path = Path(onto_config.path)
        if not path.is_absolute():
            path = base_path / path
        ontology_configs.append((path, onto_config))

    # Detect device and set batch size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE_GPU if device == "cuda" else DEFAULT_BATCH_SIZE_CPU

    if verbose:
        print(f"Device: {device}")
        print(f"Batch size: {batch_size}")
        print(f"Chunk size: {chunk_size}")
        print(f"Workers: {workers}")
        print()

    # Load embedding model with GPU support and FP16 for memory efficiency
    if verbose:
        print(f"Loading embedding model: {embedding_model}")

    # Clear GPU memory before loading
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    model = SentenceTransformer(embedding_model, device=device)

    # Use half precision on GPU to reduce memory
    if device == "cuda":
        model = model.half()

    # Collect all terms and track metadata
    all_terms = []
    stats = {}
    metadata_list: list[OntologyMetadata] = []

    # Build path -> config lookup for parallel processing
    path_to_config = {path: onto_config for path, onto_config in ontology_configs}
    paths = [path for path, _ in ontology_configs]

    if len(paths) > 1 and workers > 1:
        # Parallel loading
        if verbose:
            print(f"Loading {len(paths)} ontologies in parallel...")

        with ProcessPoolExecutor(max_workers=min(len(paths), workers)) as executor:
            futures = {executor.submit(load_ontology, p): p for p in paths}
            for future in as_completed(futures):
                path = futures[future]
                onto_config = path_to_config[path]
                try:
                    terms = future.result()
                    ontology_name = path.stem.upper()
                    stats[path.name] = len(terms)
                    all_terms.extend(terms)

                    # Store metadata
                    metadata_list.append(
                        OntologyMetadata(
                            name=ontology_name,
                            description=onto_config.description,
                            term_format=onto_config.term_format,
                            term_count=len(terms),
                        )
                    )

                    if verbose:
                        print(f"  {path.name}: {len(terms)} terms")
                except Exception as e:
                    if verbose:
                        print(f"  {path.name}: ERROR - {e}")
                    stats[path.name] = 0
    else:
        # Sequential loading (single file or workers=1)
        for path, onto_config in ontology_configs:
            if verbose:
                print(f"Loading ontology: {path}")
            try:
                terms = load_ontology(path)
                ontology_name = path.stem.upper()
                stats[path.name] = len(terms)
                all_terms.extend(terms)

                # Store metadata
                metadata_list.append(
                    OntologyMetadata(
                        name=ontology_name,
                        description=onto_config.description,
                        term_format=onto_config.term_format,
                        term_count=len(terms),
                    )
                )

                if verbose:
                    print(f"  Found {len(terms)} terms")
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                stats[path.name] = 0

    if not all_terms:
        raise ValueError("No terms found in any ontology")

    if verbose:
        print()
        print(f"Total terms to index: {len(all_terms)}")

    # Connect to LanceDB
    db = lancedb.connect(str(output_path))

    # Drop existing tables if they exist
    if "terms" in db.table_names():
        if verbose:
            print("Dropping existing terms table...")
        db.drop_table("terms")
    if "metadata" in db.table_names():
        if verbose:
            print("Dropping existing metadata table...")
        db.drop_table("metadata")

    # Store metadata table
    if verbose:
        print("Storing ontology metadata...")
    metadata_data = [m.model_dump() for m in metadata_list]
    db.create_table("metadata", metadata_data)

    # Process terms in chunks for memory efficiency and progress tracking
    total_chunks = (len(all_terms) + chunk_size - 1) // chunk_size
    table = None

    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(all_terms))
        chunk_terms = all_terms[start:end]

        if verbose:
            print(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({start}-{end})...")

        # Create embedding texts (only once per term)
        texts = [create_embedding_text(t) for t in chunk_terms]

        # Clear GPU cache before encoding chunk
        if device == "cuda":
            torch.cuda.empty_cache()

        # Encode with batching
        if verbose:
            print(f"  Encoding {len(texts)} terms...", flush=True)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Prepare data for LanceDB
        chunk_data = []
        for term, text, embedding in zip(chunk_terms, texts, embeddings):
            chunk_data.append(
                {
                    "id": term.id,
                    "ontology": term.ontology,
                    "label": term.label,
                    "synonyms": term.synonyms,
                    "definition": term.definition,
                    "text": text,
                    "vector": embedding.tolist(),
                }
            )

        # Insert chunk (create table on first chunk, append on subsequent)
        if table is None:
            table = db.create_table("terms", chunk_data)
        else:
            table.add(chunk_data)

        if verbose:
            print(f"  Inserted {len(chunk_data)} terms")

    stats["total"] = len(all_terms)

    # NOTE: IVF-PQ indexing disabled - causes accuracy issues with cosine similarity
    # Brute-force search is slower but accurate for ~30k terms
    # TODO: Investigate HNSW index as alternative

    if verbose:
        print()
        print(f"Index complete: {len(all_terms)} terms in {output_path}")

    return stats


def load_index_metadata(index_path: Path) -> list[OntologyMetadata]:
    """Load ontology metadata from an index.

    Args:
        index_path: Path to LanceDB database

    Returns:
        List of OntologyMetadata for each indexed ontology
    """
    db = lancedb.connect(str(index_path))

    if "metadata" not in db.table_names():
        return []

    table = db.open_table("metadata")
    rows = table.to_pandas().to_dict("records")

    return [OntologyMetadata(**row) for row in rows]


def update_index_metadata(
    index_path: Path,
    config: OntologiesConfig,
    verbose: bool = False,
) -> None:
    """Update metadata in an existing index without re-embedding.

    Useful for updating ontology descriptions/term_format without
    re-running the expensive embedding step.

    Args:
        index_path: Path to LanceDB database
        config: New ontologies configuration
        verbose: Print progress info
    """
    db = lancedb.connect(str(index_path))

    if "metadata" not in db.table_names():
        raise ValueError(f"No metadata table found in {index_path}")

    # Load existing metadata to get term counts
    existing = load_index_metadata(index_path)
    name_to_count = {m.name: m.term_count for m in existing}

    # Build new metadata from config
    new_metadata = []
    for onto_config in config.ontologies:
        name = Path(onto_config.path).stem.upper()
        term_count = name_to_count.get(name, 0)
        new_metadata.append(
            OntologyMetadata(
                name=name,
                description=onto_config.description,
                term_format=onto_config.term_format,
                term_count=term_count,
            )
        )

    # Drop and recreate metadata table
    if verbose:
        print(f"Updating metadata for {len(new_metadata)} ontologies...")

    db.drop_table("metadata")
    db.create_table("metadata", [m.model_dump() for m in new_metadata])

    if verbose:
        for m in new_metadata:
            print(f"  {m.name}: {m.term_count} terms")
        print("Metadata updated.")
