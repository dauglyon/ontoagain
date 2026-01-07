# OntoAgain

An experimental LLM-powered tool for ontology term identification, disambiguation, and relationship extraction from scientific papers.

## What It Does

OntoAgain extracts scientific concepts from research papers, maps them to ontology terms, and extracts relationships between them.

**Input** (plain text from a paper):
```
DNA methylation of 6mA in ciliates is associated with transcriptional activation.
```

**Output** (concepts with ontology mappings, inline in the original text):
```xml
<C n="1" q="DNA methylation; epigenetic modification">DNA methylation<M id="GO:0006306" o="GO" l="DNA methylation"/></C> of
<C n="2" q="6mA; N6-methyladenine">6mA<M id="CHEBI:21891" o="CHEBI" l="N6-methyladenine"/></C> in
<C n="3" q="ciliates; Ciliophora">ciliates<M id="NCBITaxon:5878" o="NCBITAXON" l="Ciliophora"/></C> is associated with
<C n="4" q="transcriptional activation">transcriptional activation<M id="GO:0045893" o="GO" l="positive regulation of transcription"/></C>.
```

**Extracted relationships** (optional, can be mapped to a relationship ontology):
```xml
<relations>
<R s="GO:0006306" o="GO:0045893" p="associated_with" e="1,4">DNA methylation is associated with transcriptional activation in ciliates</R>
</relations>
```

## Approach

OntoAgain uses **retrieval-augmented grounding**: vector search retrieves candidate ontology terms, then an LLM selects the best match. This reduces hallucination risk compared to asking LLMs to generate ontology IDs directly, and scales to ontologies with millions of terms.

```
┌─────────────────┐     ┌─────────────────────┐     ┌───────────────────┐
│  Research Paper │────▶│  CONCEPT-EXTRACT    │────▶│  concepts.xml     │
│                 │     │  (LLM)              │     │  <C n="1" q=".."> │
└─────────────────┘     └─────────────────────┘     └────────┬──────────┘
                                                             │
                        ┌─────────────────────┐              ▼
                        │  Concept Index      │     ┌───────────────────┐
                        │  (LanceDB + BGE-M3) │────▶│ CONCEPT-DISAMBIG  │
                        │                     │     │ (Vector + LLM)    │
                        └─────────────────────┘     └────────┬──────────┘
                                                             │
                                                             ▼
                                                    ┌───────────────────┐
                                                    │  tagged.xml       │
                                                    │  <C><M id=".."/>  │
                                                    └────────┬──────────┘
                                                             │
                        ┌─────────────────────┐              ▼
                        │  RELATION-EXTRACT   │◀────────────────────────┘
                        │  (LLM)              │
                        └────────┬────────────┘
                                 │
                                 ▼
                        ┌───────────────────┐
                        │  relations.xml    │
                        └────────┬──────────┘
                                 │
┌─────────────────────┐          ▼
│  Relation Index     │  ┌───────────────────┐
│  (optional)         │─▶│ RELATION-DISAMBIG │
└─────────────────────┘  │ (Vector + LLM)    │
                         └───────────────────┘
```

The pipeline is split into phases so each can use different models and the grounding step can scale independently. Concepts are annotated inline, preserving document structure and positions for downstream processing.

## Benchmarks

### BC5CDR: Chemical-Induced-Disease Relation Extraction

Tested on the BioCreative V CDR corpus (Chemical-Disease Relations). This is an early benchmark—results will likely improve with further prompt tuning.

| Metric | OntoAgain |
|--------|-----------|
| **F1 Score** | **60.5%** |
| Precision | 56.1% |
| Recall | 65.7% |

*20 documents from test split, 37 gold relations*

See [Running Benchmarks](#running-benchmarks) for setup instructions.

## Installation

```bash
git clone https://github.com/dauglyon/ontoagain
cd ontoagain
uv sync
```

## Quick Start

### 1. Recommend ontologies for your paper

```bash
onto recommend-ontologies paper.txt --verbose
```

### 2. Build an ontology index

Download ontologies to the data folder:

```bash
mkdir -p data/ontologies
curl -Lo data/ontologies/go.obo http://purl.obolibrary.org/obo/go.obo
curl -Lo data/ontologies/chebi.obo http://purl.obolibrary.org/obo/chebi.obo
```

Edit `sample_config.yaml` to point to your ontologies, then build the index:

```bash
onto index sample_config.yaml -O data/indexes/my-index -v
```

The config file contains descriptions that guide the LLM during disambiguation. See `sample_config.yaml` for the format.

### 3. Run the pipeline

```bash
# Step 1: Extract concepts
onto concept-extract paper.txt --index data/indexes/my-index -o data/concepts.xml -v

# Step 2: Match concepts to ontology terms
onto concept-disambiguate data/concepts.xml --index data/indexes/my-index -o data/tagged.xml -v

# Step 3: Extract relationships (optional)
onto relation-extract data/tagged.xml -o data/relations.xml -v

# Step 4: Map predicates to a relationship ontology (optional)
onto relation-disambiguate data/relations.xml --rel-index data/indexes/rel-index -o data/relations-mapped.xml -v
```

## Commands

| Command | Description |
|---------|-------------|
| `onto recommend-ontologies <paper>` | Suggest ontologies for a paper |
| `onto index <config.yaml> -O <output>` | Build vector index from ontologies |
| `onto update-metadata <config.yaml> -i <index>` | Update index metadata without re-embedding |
| `onto concept-extract <paper> [--index <path>]` | Extract concepts as XML |
| `onto concept-disambiguate <xml> --index <path>` | Match concepts to ontology terms |
| `onto relation-extract <xml>` | Extract relationships between matched concepts |
| `onto relation-disambiguate <xml> --rel-index <path>` | Map predicates to relationship ontology |
| `onto benchmark -i <index> -d <data> [-r <rel-index>]` | Run BC5CDR benchmark |

## LLM Configuration

```bash
# Anthropic (recommended)
export ANTHROPIC_API_KEY=sk-ant-...

# Or OpenAI-compatible (e.g., local models)
export OPENAI_API_BASE=https://api.example.com
export OPENAI_API_KEY=your-key

# Specify model
onto concept-extract paper.txt --model anthropic/claude-sonnet
```

## Architecture

```
ontoagain/
├── cli.py           # Typer CLI
├── identify.py      # Concept extraction with chunking
├── disambiguate.py  # Ontology matching (batches concepts with overlapping candidates)
├── relate.py        # Relationship extraction and disambiguation
├── recommend.py     # Ontology recommendations
├── index.py         # LanceDB vector index
├── models.py        # Pydantic data models
├── llm.py           # LiteLLM wrapper
├── xml_utils.py     # XML parsing utilities
└── prompts/         # LLM prompt templates
```

### Key Dependencies

- **LiteLLM**: Multi-provider LLM abstraction
- **LanceDB**: Vector database
- **Sentence-Transformers**: BGE-M3 embeddings (1024-dim)
- **OAK**: OBO/OWL ontology parsing
- **Pydantic**: Data validation

## Comparison with OntoGPT

| Aspect | OntoAgain | OntoGPT |
|--------|-----------|---------|
| **Schema** | Currently schema-free | Requires LinkML templates |
| **Grounding** | Vector retrieval + LLM verification | Direct LLM grounding |
| **Scalability** | Large ontologies via vector index | Limited by context window |
| **Output** | Inline document annotation | Structured JSON/YAML/RDF |
| **Hallucination risk** | Lower (selection from candidates) | Higher (LLM generates IDs) |
| **Multi-ontology** | Single run across all indexed ontologies | Separate template per ontology |

OntoGPT is more mature and better suited for structured knowledge base population. OntoAgain is experimental and focused on document annotation with inline position tracking.

## Running Benchmarks

### BC5CDR Setup

The BC5CDR benchmark requires downloading external data files.

1. **Download BC5CDR Corpus** from BioCreative:
   ```bash
   mkdir -p data/bc5cdr
   cd data/bc5cdr
   # Download CDR_Data.zip from https://biocreative.bioinformatics.udel.edu/resources/corpora/biocreative-v-cdr-corpus/
   unzip CDR_Data.zip
   cd ../..
   ```

2. **Download MESH 2015 Vocabulary** (the corpus uses 2015 annotations):
   ```bash
   cd data/bc5cdr
   curl -O https://nlmpubs.nlm.nih.gov/projects/mesh/2015/asciimesh/d2015.bin
   curl -O https://nlmpubs.nlm.nih.gov/projects/mesh/2015/asciimesh/c2015.bin
   cd ../..
   ```

3. **Build Indexes**:
   ```bash
   # Build MESH index
   onto index benchmarks/mesh_config.yaml -O data/indexes/mesh -v

   # Build CID relationship index (optional, improves extraction)
   onto index benchmarks/cid_config.yaml -O data/indexes/cid -v
   ```

4. **Run Benchmark**:
   ```bash
   # Point to your downloaded data with -d
   onto benchmark -i data/indexes/mesh -d data/bc5cdr -n 20 -v

   # With relationship index for better extraction guidance
   onto benchmark -i data/indexes/mesh -d data/bc5cdr -r data/indexes/cid -n 20 -v
   ```

### Sample Config Files

- `sample_config.yaml` - Template for setting up your own ontologies
- `benchmarks/mesh_config.yaml` - MESH vocabulary with disambiguation guidance
- `benchmarks/cid_config.yaml` - Chemical-Induced-Disease relationship patterns

## Future Work

- Schema support for constrained extraction
- Comparison with OntoGPT on standard benchmarks
- Domain-specific embedding models
- Smaller/local LLM support

## References

- [OntoGPT](https://github.com/monarch-initiative/ontogpt) - Schema-driven LLM extraction
- [BERN2](https://github.com/dmis-lab/BERN2) - Neural biomedical NER
- [NCBO Annotator](https://bioportal.bioontology.org/annotator) - Syntactic ontology matching
- [text2term](https://github.com/ccb-hms/text2term) - Term mapping toolkit

## License

MIT
