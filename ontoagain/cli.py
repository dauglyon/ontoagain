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
from ontoagain.index import build_index_from_config, load_index_metadata
from ontoagain.models import OntologiesConfig
from ontoagain.recommend import recommend

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


@app.command(name="identify")
def identify_cmd(
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
        concept_count = xml_output.count("<C ") + xml_output.count("<concept ")
        typer.echo(f"Extracted {concept_count} concepts")
        typer.echo("")

    # Output as XML
    if output:
        output.write_text(xml_output)
        if verbose:
            typer.echo(f"XML saved to: {output}")
    else:
        typer.echo(xml_output)


@app.command(name="disambiguate")
def disambiguate_cmd(
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
