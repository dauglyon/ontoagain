# OntoAgain

LLM-powered ontology term identification and disambiguation for scientific papers.

## Overview

OntoAgain extracts scientific concepts from research papers and maps them to ontology terms using a two-phase pipeline:

1. **IDENTIFY** - Uses an LLM to extract scientific concepts (genes, proteins, processes, species, etc.) from paper text
2. **DISAMBIGUATE** - Matches concepts to ontology terms using vector similarity search + LLM verification

Additionally, OntoAgain can **recommend ontologies** for a paper by analyzing its concepts and suggesting relevant OBO Foundry ontologies.

## Installation

```bash
# Clone and install
git clone <repo-url>
cd ontoagain
uv sync
```

## Quick Start

### 1. Recommend ontologies for your paper

```bash
onto recommend-ontologies paper.txt --verbose
```

This extracts concepts and suggests which ontologies to use (GO, CHEBI, CL, etc.).

### 2. Build an ontology index

Download ontology files from OBO Foundry and create a config file:

```bash
# Download ontologies
mkdir ontologies
curl -Lo ontologies/go.obo http://purl.obolibrary.org/obo/go.obo
curl -Lo ontologies/chebi.obo http://purl.obolibrary.org/obo/chebi.obo
```

Create `ontologies.yaml` with descriptions and term format hints:

```yaml
ontologies:
  - path: ontologies/go.obo
    description: "Biological processes, molecular functions, and cellular components"
    term_format: "Lowercase phrases like 'regulation of transcription', 'protein kinase activity'"

  - path: ontologies/chebi.obo
    description: "Chemical entities including small molecules, drugs, and metabolites"
    term_format: "Chemical names like 'ethanol', 'adenosine triphosphate', '5-methylcytosine'"

  - path: ontologies/ncbitaxon.obo
    description: "Taxonomic classification - use ONLY for species and organism names"
    term_format: "Binomial nomenclature like 'Homo sapiens', 'Escherichia coli'"

  - path: ontologies/pr.obo
    description: "Protein entities including specific proteins and protein families"
    term_format: "Protein names like 'tumor protein p53', 'DNA methyltransferase 1'"
```

Build the index:

```bash
onto index ontologies.yaml -O my-index -v
```

The config provides **grounding** - descriptions help the LLM generate better search terms and avoid mismatches (e.g., not matching concepts to NCBITaxon unless they're actually species).

### 3. Tag your paper (two-step pipeline)

```bash
# Step 1: Extract concepts (outputs XML)
onto identify paper.txt --index my-index -o concepts.xml -v

# Step 2: Match concepts to ontology terms
onto disambiguate concepts.xml --index my-index -o tagged.xml -v
```

The pipeline uses XML format. After IDENTIFY:

```xml
<concept context="programmed cell death" search="apoptosis; programmed cell death">apoptosis</concept>
```

After DISAMBIGUATE, ontology matches are added as nested elements:

```xml
<concept context="programmed cell death" search="apoptosis; programmed cell death">apoptosis
  <match ontology="GO" id="GO:0006915" label="apoptotic process"/>
  <match ontology="GO" id="GO:0008219" label="cell death"/>
</concept>
```

## Commands

| Command | Description |
|---------|-------------|
| `onto recommend-ontologies <paper>` | Suggest ontologies for a paper |
| `onto index <config.yaml> -O <output>` | Build vector index from config file |
| `onto identify <paper> [--index <path>]` | Extract concepts as XML (grounding improves results) |
| `onto disambiguate <concepts.xml> --index <path>` | Match concepts to ontology terms |
| `onto stats <results.json>` | Show statistics from results |

## Ontology Config File

The config file (`ontologies.yaml`) defines which ontologies to index and provides metadata for grounding:

```yaml
ontologies:
  - path: ontologies/go.obo          # Path to OBO/OWL file
    description: "What this ontology contains"  # Helps LLM understand scope
    term_format: "How terms are typically phrased"  # Helps generate search terms
```

**Why grounding matters**: Without descriptions, the LLM might match "transposable element" to a NCBITaxon entry (bacteria that have transposons) instead of recognizing it's a concept. The description "use ONLY for species and organism names" prevents this.

## LLM Configuration

Set environment variables for your LLM provider:

```bash
# OpenAI-compatible API (e.g., CBORG, local models)
export OPENAI_API_BASE=https://api.example.com
export OPENAI_API_KEY=your-key

# Or direct Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

Use `--model` to specify the model:

```bash
onto identify paper.txt --model anthropic/claude-sonnet -o concepts.xml
onto disambiguate concepts.xml --index my-index --model anthropic/claude-sonnet
```

## How It Works

### Concept Extraction (IDENTIFY)

The LLM extracts scientific concepts as XML `<concept>` tags with attributes:
- **text** (tag content): The surface form in the paper
- **context**: Resolved meaning (expanded abbreviations, synonyms)
- **search**: Terms for ontology matching

When an index is provided, ontology metadata is injected into the prompt to help generate better search terms that match the ontology vocabulary.

### Ontology Matching (DISAMBIGUATE)

1. Vector search (BGE-M3 embeddings, IVF-PQ indexed) retrieves candidate terms from LanceDB
2. Concepts with overlapping candidates are batched together for efficiency
3. LLM selects the best matching term(s) for each concept
4. XML is updated with nested `<match/>` elements containing `ontology`, `id`, and `label`

Ontology descriptions from the config help the LLM reject inappropriate matches (e.g., not matching "transposable element" to NCBITaxon just because there's a taxonomic entry with that name).

### Ontology Recommendations

Analyzes extracted concepts and suggests relevant OBO Foundry ontologies based on concept types (chemicals, processes, species, etc.).

## Architecture

```
ontoagain/
├── cli.py          # Typer CLI commands
├── identify.py     # Concept extraction
├── disambiguate.py # Ontology matching with batching
├── recommend.py    # Ontology recommendations
├── index.py        # LanceDB vector index
├── models.py       # Pydantic data models
├── llm.py          # LiteLLM wrapper
├── xml_utils.py    # XML parsing utilities
└── prompts/        # LLM prompt templates
```

## Development

```bash
# Install with dev dependencies
uv sync --dev

# Run tests
uv run pytest
```

## License

MIT
