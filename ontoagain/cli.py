"""CLI for OntoAgain - ontology term tagging for scientific papers."""

import json
import os
from pathlib import Path
from typing import Annotated, Optional

from dotenv import load_dotenv
import typer

# Load .env file if present
load_dotenv()

import yaml

from ontoagain.disambiguate import disambiguate
from ontoagain.identify import identify
from ontoagain.index import build_index_from_config, load_index_metadata, update_index_metadata
from ontoagain.models import OntologiesConfig
from ontoagain.recommend import recommend
from ontoagain.relate import relate_extract, relate_disambiguate, raw_relationships_to_xml, relationships_to_xml

app = typer.Typer(
    name="onto",
    help="Tag scientific papers with ontology terms.",
    no_args_is_help=True,
    rich_markup_mode=None,
)


@app.command()
def index(
    config: Annotated[
        Path,
        typer.Argument(help="Path to ontologies config YAML file"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-O", help="Output directory for index"),
    ],
    batch_size: Annotated[
        Optional[int],
        typer.Option("--batch-size", "-b", help="Batch size for encoding (auto-detected if not set)"),
    ] = None,
    chunk_size: Annotated[
        int,
        typer.Option("--chunk-size", "-c", help="Terms per processing chunk"),
    ] = 50000,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Parallel workers for ontology parsing"),
    ] = 4,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show progress info"),
    ] = False,
):
    """Build an ontology index from a config file.

    The config file (YAML) specifies ontologies with their descriptions
    and term format guidelines. This metadata is stored in the index
    and used to improve concept identification and disambiguation.

    Example config:

        ontologies:
          - path: ontologies/go.obo
            description: "Biological processes, molecular functions, cellular components"
            term_format: "Lowercase phrases like 'regulation of transcription'"
    """
    if not config.exists():
        typer.echo(f"Error: Config file not found: {config}", err=True)
        raise typer.Exit(1)

    # Parse config
    try:
        config_data = yaml.safe_load(config.read_text())
        ontologies_config = OntologiesConfig(**config_data)
    except Exception as e:
        typer.echo(f"Error parsing config: {e}", err=True)
        raise typer.Exit(1)

    if verbose:
        typer.echo(f"Loaded config with {len(ontologies_config.ontologies)} ontologies")

    # Build index with metadata
    try:
        stats = build_index_from_config(
            ontologies_config,
            output,
            base_path=config.parent,
            batch_size=batch_size,
            chunk_size=chunk_size,
            workers=workers,
            verbose=verbose,
        )
        typer.echo(f"Index created at: {output}")
        typer.echo(f"Total terms indexed: {stats['total']}")
        for name, count in stats.items():
            if name != "total":
                typer.echo(f"  {name}: {count}")
    except Exception as e:
        typer.echo(f"Error building index: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="update-metadata")
def update_metadata_cmd(
    config: Annotated[
        Path,
        typer.Argument(help="Path to ontologies config YAML file"),
    ],
    index_path: Annotated[
        Path,
        typer.Option("--index", "-i", help="Path to existing index"),
    ],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show progress info"),
    ] = False,
):
    """Update index metadata without re-embedding.

    Updates the ontology descriptions and term_format from the config
    without re-running the expensive embedding step. Useful when you've
    only changed the config descriptions.
    """
    if not config.exists():
        typer.echo(f"Error: Config file not found: {config}", err=True)
        raise typer.Exit(1)

    if not index_path.exists():
        typer.echo(f"Error: Index not found: {index_path}", err=True)
        raise typer.Exit(1)

    # Parse config
    try:
        config_data = yaml.safe_load(config.read_text())
        ontologies_config = OntologiesConfig(**config_data)
    except Exception as e:
        typer.echo(f"Error parsing config: {e}", err=True)
        raise typer.Exit(1)

    # Update metadata
    try:
        update_index_metadata(index_path, ontologies_config, verbose=verbose)
        typer.echo(f"Metadata updated in: {index_path}")
    except Exception as e:
        typer.echo(f"Error updating metadata: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def recommend_ontologies(
    paper: Annotated[
        Path,
        typer.Argument(help="Path to paper text file"),
    ],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="LLM model name"),
    ] = "anthropic/claude-sonnet",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show progress info"),
    ] = False,
):
    """Recommend ontologies for a paper based on its content.

    Extracts concepts from the paper and suggests relevant OBO Foundry
    ontologies that could be used for annotation.
    """
    if verbose:
        typer.echo("=== OntoAgain Recommend ===")
        typer.echo(f"OPENAI_API_BASE: {os.environ.get('OPENAI_API_BASE', '(not set)')}")
        typer.echo(f"OPENAI_API_KEY: {'(set)' if os.environ.get('OPENAI_API_KEY') else '(not set)'}")
        typer.echo(f"Model: {model}")
        typer.echo("")

    # Validate inputs
    if not paper.exists():
        typer.echo(f"Error: Paper file not found: {paper}", err=True)
        raise typer.Exit(1)

    # Read paper
    text = paper.read_text()

    if verbose:
        typer.echo(f"Paper: {paper}")
        typer.echo(f"Paper length: {len(text)} characters")
        typer.echo("")

    # Run recommendation
    recommendations, concepts = recommend(text, model=model, verbose=verbose)

    # Output results
    typer.echo("")
    typer.echo("=" * 60)
    typer.echo(f"Extracted {len(concepts)} concepts from paper")
    typer.echo("=" * 60)
    typer.echo("")

    if not recommendations:
        typer.echo("No ontology recommendations generated.")
        return

    typer.echo(f"Recommended Ontologies ({len(recommendations)}):")
    typer.echo("-" * 60)

    for i, rec in enumerate(recommendations, 1):
        typer.echo(f"\n{i}. {rec.id} - {rec.name}")
        typer.echo(f"   Relevance: {rec.relevance}")
        if rec.example_concepts:
            typer.echo(f"   Example concepts: {', '.join(rec.example_concepts[:5])}")
        if rec.download_url:
            typer.echo(f"   Download: {rec.download_url}")

    typer.echo("")
    typer.echo("-" * 60)
    typer.echo("To build an index with these ontologies:")
    typer.echo("  1. Download the .obo files from the URLs above")
    typer.echo("  2. Run: onto index -o file1.obo -o file2.obo -O my-index")
    typer.echo("  3. Tag your paper: onto tag paper.txt --index my-index")


@app.command(name="concept-extract")
def concept_extract_cmd(
    paper: Annotated[
        Path,
        typer.Argument(help="Path to paper text file"),
    ],
    index_path: Annotated[
        Optional[Path],
        typer.Option("--index", "-i", help="Path to ontology index (for grounding)"),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="LLM model name"),
    ] = "anthropic/claude-sonnet",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output XML file (default: stdout)"),
    ] = None,
    max_concurrent: Annotated[
        int,
        typer.Option("--max-concurrent", "-c", help="Max concurrent LLM calls (1=sequential)"),
    ] = 4,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show progress info"),
    ] = False,
):
    """Extract concepts from a paper (IDENTIFY step only).

    Outputs XML with <concept> tags for later disambiguation.
    Optionally accepts an index path to load ontology metadata for better grounding.
    """
    if verbose:
        typer.echo("=== OntoAgain IDENTIFY ===")
        typer.echo(f"Model: {model}")
        typer.echo("")

    if not paper.exists():
        typer.echo(f"Error: Paper file not found: {paper}", err=True)
        raise typer.Exit(1)

    # Load ontology metadata if index provided
    ontology_metadata = None
    if index_path and index_path.exists():
        ontology_metadata = load_index_metadata(index_path)
        if verbose and ontology_metadata:
            typer.echo(f"Loaded metadata for {len(ontology_metadata)} ontologies")

    text = paper.read_text()

    if verbose:
        typer.echo(f"Paper: {paper}")
        typer.echo(f"Paper length: {len(text)} characters")
        typer.echo("")

    # Run IDENTIFY
    if verbose:
        typer.echo("Extracting concepts...")
    xml_output = identify(
        text, model=model, ontology_metadata=ontology_metadata, verbose=verbose,
        max_concurrent=max_concurrent,
    )

    # Count concepts for verbose output
    if verbose:
        concept_count = xml_output.count("<C n=")
        typer.echo(f"Extracted {concept_count} concepts")
        typer.echo("")

    # Output as XML
    if output:
        output.write_text(xml_output)
        if verbose:
            typer.echo(f"XML saved to: {output}")
    else:
        typer.echo(xml_output)


@app.command(name="concept-disambiguate")
def concept_disambiguate_cmd(
    xml_file: Annotated[
        Path,
        typer.Argument(help="Path to XML file with concept tags (from identify command)"),
    ],
    index_path: Annotated[
        Path,
        typer.Option("--index", "-i", help="Path to ontology index"),
    ],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="LLM model name"),
    ] = "anthropic/claude-sonnet",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output XML file (default: stdout)"),
    ] = None,
    max_concurrent: Annotated[
        int,
        typer.Option("--max-concurrent", "-c", help="Max concurrent LLM calls (1=sequential)"),
    ] = 6,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show progress info"),
    ] = False,
):
    """Match concepts to ontology terms (DISAMBIGUATE step).

    Takes XML with <concept> tags from identify command and adds ontology matches.
    """
    if verbose:
        typer.echo("=== OntoAgain DISAMBIGUATE ===")
        typer.echo(f"Model: {model}")
        typer.echo(f"Index: {index_path}")
        typer.echo("")

    if not xml_file.exists():
        typer.echo(f"Error: XML file not found: {xml_file}", err=True)
        raise typer.Exit(1)

    if not index_path.exists():
        typer.echo(f"Error: Index not found: {index_path}", err=True)
        raise typer.Exit(1)

    # Load ontology metadata from index
    ontology_metadata = load_index_metadata(index_path)
    if verbose and ontology_metadata:
        typer.echo(f"Loaded metadata for {len(ontology_metadata)} ontologies")

    # Load XML
    xml_input = xml_file.read_text()

    if verbose:
        concept_count = xml_input.count("<C ") + xml_input.count("<concept ")
        typer.echo(f"Loaded {concept_count} concepts from {xml_file}")
        typer.echo("")

    # Run DISAMBIGUATE
    if verbose:
        typer.echo("Matching concepts to ontology...")
    updated_xml, stats = disambiguate(
        xml_input, index_path, model=model, verbose=verbose,
        ontology_metadata=ontology_metadata, max_concurrent=max_concurrent,
    )

    if verbose:
        typer.echo(f"DISAMBIGUATE complete.")
        typer.echo(f"  LLM batches: {stats.get('batches', 'n/a')}")
        typer.echo(f"  Matched: {stats['matched']}")
        typer.echo(f"  Unmatched: {stats['unmatched']}")
        typer.echo("")

    # Output
    if output:
        output.write_text(updated_xml)
        if verbose:
            typer.echo(f"Results written to: {output}")
    else:
        typer.echo(updated_xml)


@app.command(name="relation-extract")
def relation_extract_cmd(
    xml_file: Annotated[
        Path,
        typer.Argument(help="Path to disambiguated XML file with <M/> tags"),
    ],
    rel_index: Annotated[
        Optional[Path],
        typer.Option("--rel-index", "-r", help="Path to relationship ontology index (for filtering)"),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="LLM model name"),
    ] = "anthropic/claude-sonnet",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output XML file (default: stdout)"),
    ] = None,
    max_concurrent: Annotated[
        int,
        typer.Option("--max-concurrent", "-c", help="Max concurrent LLM calls"),
    ] = 4,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show progress info"),
    ] = False,
):
    """Extract relationships from disambiguated concepts.

    Takes XML with <C n="..."><M id="..."/></C> tags and identifies
    relationships between the matched concepts.

    Output is XML with <R s="..." o="..." p="..." e="...">summary</R> elements.
    """
    if verbose:
        typer.echo("=== OntoAgain RELATION-EXTRACT ===")
        typer.echo(f"Model: {model}")
        if rel_index:
            typer.echo(f"Relationship index: {rel_index}")
        typer.echo("")

    if not xml_file.exists():
        typer.echo(f"Error: XML file not found: {xml_file}", err=True)
        raise typer.Exit(1)

    # Load relationship ontology metadata if provided
    relationship_metadata = None
    if rel_index and rel_index.exists():
        relationship_metadata = load_index_metadata(rel_index)
        if verbose and relationship_metadata:
            typer.echo(f"Loaded relationship metadata for {len(relationship_metadata)} ontologies")

    # Load XML
    xml_input = xml_file.read_text()

    if verbose:
        typer.echo(f"Loaded XML from {xml_file}")
        typer.echo("")

    # Run RELATION-EXTRACT
    if verbose:
        typer.echo("Extracting relationships...")
    relationships = relate_extract(
        xml_input, model=model, verbose=verbose, max_concurrent=max_concurrent,
        relationship_metadata=relationship_metadata,
    )

    if verbose:
        typer.echo(f"Extracted {len(relationships)} relationships")
        typer.echo("")

    # Output as XML
    xml_output = raw_relationships_to_xml(relationships)

    if output:
        output.write_text(xml_output)
        if verbose:
            typer.echo(f"Results written to: {output}")
    else:
        typer.echo(xml_output)


@app.command(name="relation-disambiguate")
def relation_disambiguate_cmd(
    relations_file: Annotated[
        Path,
        typer.Argument(help="Path to XML file with <R/> elements from relation-extract"),
    ],
    relationship_index: Annotated[
        Path,
        typer.Option("--rel-index", "-r", help="Path to relationship ontology index (e.g., RO)"),
    ],
    concept_index: Annotated[
        Optional[Path],
        typer.Option("--concept-index", "-i", help="Optional path to concept index for context"),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="LLM model name"),
    ] = "anthropic/claude-sonnet",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output XML file (default: stdout)"),
    ] = None,
    max_concurrent: Annotated[
        int,
        typer.Option("--max-concurrent", "-c", help="Max concurrent LLM calls"),
    ] = 6,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show progress info"),
    ] = False,
):
    """Disambiguate relationship predicates to ontology terms.

    Takes XML with <R/> elements and maps predicates to relationship
    ontology terms (e.g., RO - Relation Ontology).

    Output is XML with predicate IDs added (p= attribute updated).
    """
    if verbose:
        typer.echo("=== OntoAgain RELATION-DISAMBIGUATE ===")
        typer.echo(f"Model: {model}")
        typer.echo(f"Relationship index: {relationship_index}")
        if concept_index:
            typer.echo(f"Concept index: {concept_index}")
        typer.echo("")

    if not relations_file.exists():
        typer.echo(f"Error: Relations file not found: {relations_file}", err=True)
        raise typer.Exit(1)

    if not relationship_index.exists():
        typer.echo(f"Error: Relationship index not found: {relationship_index}", err=True)
        raise typer.Exit(1)

    # Load ontology metadata from relationship index
    ontology_metadata = load_index_metadata(relationship_index)
    if verbose and ontology_metadata:
        typer.echo(f"Loaded metadata for {len(ontology_metadata)} ontologies")

    # Parse XML to get RawRelationship objects
    from ontoagain.models import RawRelationship
    from ontoagain.xml_utils import parse_xml_fragment

    xml_text = relations_file.read_text()
    root = parse_xml_fragment(xml_text)
    relationships = []

    for elem in root.findall(".//R"):
        evidence_ids = elem.get("e", "")
        concept_ids = [x.strip() for x in evidence_ids.split(",") if x.strip()]
        relationships.append(RawRelationship(
            subject_id=elem.get("s", ""),
            object_id=elem.get("o", ""),
            predicate=elem.get("p", ""),
            concept_ids=concept_ids,
            summary=elem.text.strip() if elem.text else "",
        ))

    if verbose:
        typer.echo(f"Loaded {len(relationships)} relationships from {relations_file}")
        typer.echo("")

    # Run RELATION-DISAMBIGUATE
    if verbose:
        typer.echo("Disambiguating predicates...")
    disambiguated = relate_disambiguate(
        relationships,
        relationship_index,
        model=model,
        concept_index=concept_index,
        verbose=verbose,
        max_concurrent=max_concurrent,
        ontology_metadata=ontology_metadata,
    )

    if verbose:
        matched = sum(1 for r in disambiguated if r.predicate_id)
        typer.echo(f"Disambiguated: {matched}/{len(disambiguated)} predicates matched")
        typer.echo("")

    # Output as XML
    xml_output = relationships_to_xml(disambiguated)

    if output:
        output.write_text(xml_output)
        if verbose:
            typer.echo(f"Results written to: {output}")
    else:
        typer.echo(xml_output)


@app.command(name="benchmark")
def benchmark_cmd(
    index_path: Annotated[
        Path,
        typer.Option("--index", "-i", help="Path to concept index (e.g., MESH)"),
    ],
    data_dir: Annotated[
        Path,
        typer.Option("--data", "-d", help="Path to benchmark data directory (contains CDR_Data/)"),
    ] = Path("benchmarks/data"),
    rel_index: Annotated[
        Optional[Path],
        typer.Option("--rel-index", "-r", help="Path to relationship index (e.g., CID)"),
    ] = None,
    split: Annotated[
        str,
        typer.Option("--split", "-s", help="Dataset split (train, dev, test)"),
    ] = "test",
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="LLM model name"),
    ] = "anthropic/claude-sonnet",
    limit: Annotated[
        Optional[int],
        typer.Option("--limit", "-n", help="Limit number of documents to process"),
    ] = None,
    pmids: Annotated[
        Optional[str],
        typer.Option("--pmid", "-p", help="Comma-separated list of specific PMIDs to process"),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output JSON file for detailed results"),
    ] = None,
    max_concurrent: Annotated[
        int,
        typer.Option("--max-concurrent", "-c", help="Max concurrent LLM calls"),
    ] = 4,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show progress info"),
    ] = False,
):
    """Run BC5CDR relation extraction benchmark.

    Evaluates the full pipeline (IDENTIFY → DISAMBIGUATE → RELATE-EXTRACT)
    against the BC5CDR Chemical-Disease Relation corpus.

    Requires a MESH index for concept disambiguation and BC5CDR data files.
    """
    from benchmarks.bc5cdr import load_pubtator, get_corpus_path, evaluate_relations

    if not index_path.exists():
        typer.echo(f"Error: Index not found: {index_path}", err=True)
        typer.echo("Build MESH index with: onto index benchmarks/mesh_config.yaml -O data/indexes/mesh")
        raise typer.Exit(1)

    if not data_dir.exists():
        typer.echo(f"Error: Data directory not found: {data_dir}", err=True)
        typer.echo("Download BC5CDR corpus to benchmarks/data/")
        raise typer.Exit(1)

    if verbose:
        typer.echo("=== BC5CDR Benchmark (Full Pipeline) ===")
        typer.echo(f"Split: {split}")
        typer.echo(f"Model: {model}")
        typer.echo(f"Index: {index_path}")
        typer.echo("")

    # Load documents
    corpus_path = get_corpus_path(split, data_dir)
    if not corpus_path.exists():
        typer.echo(f"Error: Corpus file not found: {corpus_path}", err=True)
        typer.echo(f"Expected BC5CDR data at: {data_dir}/CDR_Data/CDR.Corpus.v010516/")
        raise typer.Exit(1)
    documents = load_pubtator(corpus_path)

    # Filter by specific PMIDs if provided
    if pmids:
        pmid_set = {p.strip() for p in pmids.split(",")}
        documents = [d for d in documents if d.pmid in pmid_set]
        if not documents:
            typer.echo(f"Error: No documents found with PMIDs: {pmids}", err=True)
            raise typer.Exit(1)

    if limit:
        documents = documents[:limit]

    if verbose:
        total_rels = sum(len(d.relations) for d in documents)
        typer.echo(f"Loaded {len(documents)} documents with {total_rels} gold relations")
        typer.echo("")

    # Load ontology metadata
    ontology_metadata = load_index_metadata(index_path)
    if verbose and ontology_metadata:
        typer.echo(f"Loaded metadata for {len(ontology_metadata)} ontologies")

    # Load relationship metadata if provided
    rel_metadata = None
    if rel_index and rel_index.exists():
        rel_metadata = load_index_metadata(rel_index)
        if verbose:
            typer.echo(f"Loaded relationship metadata from {rel_index}")
    typer.echo("")

    # Process documents
    all_predicted: list[tuple[str, str]] = []
    all_gold = []
    results_list = []

    for i, doc in enumerate(documents):
        if verbose:
            typer.echo(f"[{i+1}/{len(documents)}] Processing PMID {doc.pmid}...")

        try:
            # Step 1: IDENTIFY - extract concepts from raw text
            if verbose:
                typer.echo("    IDENTIFY...")
            identified_xml = identify(
                doc.text,
                model=model,
                ontology_metadata=ontology_metadata,
                verbose=False,
                max_concurrent=max_concurrent,
            )

            # Step 2: DISAMBIGUATE - match concepts to MESH
            if verbose:
                typer.echo("    DISAMBIGUATE...")
            disambiguated_xml, disamb_stats = disambiguate(
                identified_xml,
                index_path,
                model=model,
                ontology_metadata=ontology_metadata,
                verbose=False,
                max_concurrent=max_concurrent,
            )

            # Step 3: RELATE-EXTRACT - extract relationships
            if verbose:
                typer.echo("    RELATE-EXTRACT...")
            relationships = relate_extract(
                disambiguated_xml,
                model=model,
                verbose=False,
                max_concurrent=max_concurrent,
                relationship_metadata=rel_metadata,
            )

            # Extract CID predictions (chemical -> disease)
            # For BC5CDR, the LLM extracts "induces" relationships which are chemical->disease
            # We trust the subject/object order from the LLM
            doc_predicted = []
            for rel in relationships:
                # Use the relationship as extracted (subject -> object)
                doc_predicted.append((rel.subject_id, rel.object_id))

            # Deduplicate
            normalized_predicted = list(set(doc_predicted))

            all_predicted.extend(normalized_predicted)
            all_gold.extend(doc.relations)

            doc_result = {
                "pmid": doc.pmid,
                "predicted": normalized_predicted,
                "gold": [(r.chemical_id, r.disease_id) for r in doc.relations],
                "concepts_matched": disamb_stats.get("matched", 0),
                "concepts_unmatched": disamb_stats.get("unmatched", 0),
                "relationships_extracted": len(relationships),
            }
            doc_metrics = evaluate_relations(normalized_predicted, doc.relations)
            doc_result.update(doc_metrics)
            results_list.append(doc_result)

            if verbose:
                typer.echo(
                    f"    Concepts: {disamb_stats.get('matched', 0)} matched, "
                    f"Relations: {len(relationships)} extracted, "
                    f"CID: {len(normalized_predicted)} predicted vs {len(doc.relations)} gold"
                )
                typer.echo(
                    f"    P={doc_metrics['precision']:.2f} R={doc_metrics['recall']:.2f} F1={doc_metrics['f1']:.2f}"
                )

        except Exception as e:
            if verbose:
                typer.echo(f"    Error: {e}")
            results_list.append({
                "pmid": doc.pmid,
                "error": str(e),
                "gold": [(r.chemical_id, r.disease_id) for r in doc.relations],
            })
            all_gold.extend(doc.relations)

    # Compute overall metrics
    from benchmarks.bc5cdr import Relation
    gold_relations = [Relation(chemical_id=c, disease_id=d) for c, d in
                      [(r.chemical_id, r.disease_id) for r in all_gold]]
    overall_metrics = evaluate_relations(all_predicted, gold_relations)

    # Print summary
    typer.echo("")
    typer.echo("=" * 60)
    typer.echo("BC5CDR Benchmark Results (Full Pipeline)")
    typer.echo("=" * 60)
    typer.echo(f"Documents: {len(documents)}")
    typer.echo(f"Gold relations: {len(all_gold)}")
    typer.echo(f"Predicted relations: {len(all_predicted)}")
    typer.echo("")
    typer.echo(f"Precision: {overall_metrics['precision']:.4f}")
    typer.echo(f"Recall:    {overall_metrics['recall']:.4f}")
    typer.echo(f"F1:        {overall_metrics['f1']:.4f}")
    typer.echo("=" * 60)

    # Save detailed results
    if output:
        result_data = {
            "split": split,
            "model": model,
            "index": str(index_path),
            "documents": len(documents),
            "metrics": overall_metrics,
            "per_document": results_list,
        }
        output.write_text(json.dumps(result_data, indent=2))
        if verbose:
            typer.echo(f"Detailed results saved to: {output}")


@app.command()
def stats(
    results: Annotated[
        Path,
        typer.Argument(help="Path to results JSON file"),
    ],
):
    """Show statistics from a results file."""
    if not results.exists():
        typer.echo(f"Error: Results file not found: {results}", err=True)
        raise typer.Exit(1)

    data = json.loads(results.read_text())

    typer.echo("OntoAgain Results Summary")
    typer.echo("=" * 40)

    stats_data = data.get("stats", {})
    typer.echo(f"Total concepts:    {stats_data.get('total', 0)}")
    typer.echo(f"Matched:           {stats_data.get('matched', 0)}")
    typer.echo(f"Unmatched:         {stats_data.get('unmatched', 0)}")
    typer.echo(f"Total mappings:    {stats_data.get('total_mappings', 0)}")

    # Show unmatched concepts
    concepts = data.get("concepts", [])
    unmatched = [c for c in concepts if not c.get("matches")]
    if unmatched:
        typer.echo(f"\nUnmatched concepts ({len(unmatched)}):")
        for c in unmatched[:10]:  # Show first 10
            typer.echo(f"  - {c['text']} ({c.get('context', '')})")
        if len(unmatched) > 10:
            typer.echo(f"  ... and {len(unmatched) - 10} more")


if __name__ == "__main__":
    app()
