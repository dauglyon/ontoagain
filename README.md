# OntoAgain

LLM-powered ontology term identification and disambiguation for scientific papers.

## What It Does

OntoAgain extracts scientific concepts from research papers and maps them to ontology terms.

**Input** (plain text from a paper):
```
DNA methylation of 6mA in ciliates is associated with transcriptional activation.
```

**Output** (after IDENTIFY → DISAMBIGUATE pipeline):
```xml
<concept context="DNA methylation; epigenetic modification">
  DNA methylation
  <match ontology="GO" id="GO:0006306" label="DNA methylation"/>
</concept> of
<concept context="6mA; N6-methyladenine; adenine methylation">
  6mA
  <match ontology="CHEBI" id="CHEBI:21891" label="N6-methyladenine"/>
</concept> in
<concept context="ciliates; ciliate protozoans; Ciliophora">
  ciliates
  <match ontology="NCBITAXON" id="NCBITaxon:5878" label="Ciliophora"/>
</concept> is associated with
<concept context="transcriptional activation; gene activation">
  transcriptional activation
  <match ontology="GO" id="GO:0045893" label="positive regulation of transcription"/>
</concept>.
```

## Why OntoAgain?

### The Problem with Existing Approaches

| Tool | Approach | Limitation |
|------|----------|------------|
| **NCBO Annotator** | Syntactic string matching | No semantic understanding; misses synonyms and contextual meaning |
| **BERN2 / PubTator** | Trained neural NER models | Fixed entity types; can't use custom ontologies; requires retraining |
| **Zooma** | Dictionary lookup | No disambiguation; limited to exact/fuzzy matches |
| **text2term** | Edit distance / TF-IDF | No contextual reasoning; struggles with abbreviations |
| **OntoGPT (SPIRES)** | LLM with schema templates | Requires LinkML schema authoring; limited by context window |
| **Pure LLM** | Direct GPT/Claude queries | Hallucination risk; can't scale to large ontologies |

### OntoAgain's Approach: Retrieval-Augmented Grounding

OntoAgain combines the **semantic understanding of LLMs** with the **precision of vector retrieval**:

```
┌─────────────────┐     ┌─────────────────────┐     ┌───────────────────┐
│  Research Paper │────▶│  IDENTIFY (LLM)     │────▶│  Concepts (XML)   │
│                 │     │  Extract concepts   │     │                   │
└─────────────────┘     │  with context       │     └────────┬──────────┘
                        └─────────────────────┘              │
                                                             ▼
┌─────────────────┐     ┌─────────────────────┐     ┌───────────────────┐
│  Tagged Output  │◀────│  DISAMBIGUATE       │◀────│  Vector Index     │
│  with ontology  │     │  Retrieve candidates│     │  3M+ terms        │
│  mappings       │     │  + LLM verification │     │  (LanceDB)        │
└─────────────────┘     └─────────────────────┘     └───────────────────┘
```

## Key Innovations

### 1. Separation of Extraction and Grounding

Unlike OntoGPT which does both in a single pass, OntoAgain separates:

- **IDENTIFY**: LLM extracts concepts with rich context (expanded abbreviations, synonyms, ontology-friendly terms)
- **DISAMBIGUATE**: Vector search retrieves candidates, LLM verifies best match

This separation means each phase can use different models (fast/cheap for extraction, precise for verification) and the grounding step can scale to ontologies with millions of terms.

### 2. Retrieval-Augmented Grounding (RAG for Ontologies)

Pure LLM approaches suffer from:
- **Hallucination**: LLMs generate plausible-sounding but incorrect ontology IDs
- **Context limits**: Can't fit 3M ontology terms in a prompt
- **Reproducibility**: API-based LLMs vary between calls

OntoAgain uses **BGE-M3 embeddings + LanceDB IVF-PQ indexing** to retrieve relevant candidates, then asks the LLM to *select* rather than *generate*. This grounds outputs in real ontology terms.

### 3. Schema-Free Extraction

OntoGPT requires defining LinkML schemas that specify expected entity types and relations. OntoAgain is **schema-free**—it extracts all scientific concepts without predefined structure. This is better for:
- Exploratory annotation of unfamiliar domains
- Papers spanning multiple fields
- Quick prototyping without schema design

### 4. Inline Document Annotation

OntoAgain preserves the original document structure with inline XML tags. This maintains:
- Concept positions in text
- Reading flow for human review
- Context for downstream processing

Compare to OntoGPT's structured JSON output which loses document structure.

### 5. Batching by Candidate Overlap

A novel optimization: concepts with overlapping candidate sets are grouped into batches for a single LLM call. For 120 concepts, this reduced 120 LLM calls to 21 batches—an **83% reduction** in API calls.

### 6. Ontology Grounding Metadata

The config file includes descriptions that help the LLM avoid category errors:

```yaml
ontologies:
  - path: ontologies/ncbitaxon.obo
    description: "Taxonomic classification - use ONLY for species names, NOT biological concepts"
```

This prevents matching "transposable element" to an NCBITaxon entry for a bacterium that contains transposons.

## Performance

Tested on a Nature Genetics paper (109K characters) about DNA methylation.

| Metric | Value |
|--------|-------|
| Index size | 3.3M terms (GO + CHEBI + NCBITaxon + PR) |
| Concepts extracted (5K excerpt) | 120 |
| IDENTIFY time | 82s |
| DISAMBIGUATE time | 66s |
| Match rate | 59% (71/120) |
| LLM batch reduction | 83% (120 → 21 batches) |

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

Download ontologies and create a config:

```bash
mkdir ontologies
curl -Lo ontologies/go.obo http://purl.obolibrary.org/obo/go.obo
curl -Lo ontologies/chebi.obo http://purl.obolibrary.org/obo/chebi.obo
```

Create `ontologies.yaml`:

```yaml
ontologies:
  - path: ontologies/go.obo
    description: "Biological processes, molecular functions, and cellular components"
    term_format: "Lowercase phrases like 'regulation of transcription', 'protein kinase activity'"

  - path: ontologies/chebi.obo
    description: "Chemical entities including small molecules, drugs, and metabolites"
    term_format: "Chemical names like 'ethanol', 'adenosine triphosphate'"
```

Build the index:

```bash
onto index ontologies.yaml -O my-index -v
```

### 3. Run the pipeline

```bash
# Extract concepts
onto identify paper.txt --index my-index -o concepts.xml -v

# Match to ontology terms
onto disambiguate concepts.xml --index my-index -o tagged.xml -v
```

## Commands

| Command | Description |
|---------|-------------|
| `onto recommend-ontologies <paper>` | Suggest ontologies for a paper |
| `onto index <config.yaml> -O <output>` | Build vector index from ontologies |
| `onto identify <paper> [--index <path>]` | Extract concepts as XML |
| `onto disambiguate <xml> --index <path>` | Match concepts to ontology terms |

## LLM Configuration

```bash
# Anthropic (recommended)
export ANTHROPIC_API_KEY=sk-ant-...

# Or OpenAI-compatible (e.g., local models)
export OPENAI_API_BASE=https://api.example.com
export OPENAI_API_KEY=your-key

# Specify model
onto identify paper.txt --model anthropic/claude-sonnet
```

## Architecture

```
ontoagain/
├── cli.py           # Typer CLI
├── identify.py      # Concept extraction with chunking
├── disambiguate.py  # Ontology matching with batching
├── recommend.py     # Ontology recommendations
├── index.py         # LanceDB vector index
├── models.py        # Pydantic data models
├── llm.py           # LiteLLM wrapper
├── xml_utils.py     # XML parsing utilities
└── prompts/         # LLM prompt templates
```

### Key Dependencies

- **LiteLLM**: Multi-provider LLM abstraction
- **LanceDB**: Vector database with IVF-PQ indexing
- **Sentence-Transformers**: BGE-M3 embeddings (1024-dim)
- **OAK**: OBO/OWL ontology parsing
- **Pydantic**: Data validation

## Comparison with OntoGPT

| Aspect | OntoAgain | OntoGPT |
|--------|-----------|---------|
| **Schema** | Schema-free | Requires LinkML templates |
| **Grounding** | Vector retrieval + LLM verification | Direct LLM grounding |
| **Scalability** | 3M+ terms via vector index | Limited by context window |
| **Output** | Inline XML annotation | Structured JSON/YAML/RDF |
| **Hallucination risk** | Low (selection from candidates) | Higher (LLM generates IDs) |
| **Use case** | Document annotation | Knowledge base population |

## Future Work

### Better Embeddings
- Domain-specific models (BioLord, PubMedBERT)
- Multi-vector representations for synonyms
- Fine-tuning on ontology structure

### Parallelization
- Async LLM calls during chunking
- Concurrent batch processing
- GPU-accelerated search

### Model Optimization
- Smaller models for IDENTIFY (Haiku, local LLMs)
- Reserve larger models for DISAMBIGUATE
- Cost/quality tradeoff options

### Context Engineering
- Two-pass processing (extract abbreviations first)
- Section-aware chunking
- Coreference resolution

## References

- [OntoGPT](https://github.com/monarch-initiative/ontogpt) - Schema-driven LLM extraction
- [BERN2](https://github.com/dmis-lab/BERN2) - Neural biomedical NER
- [NCBO Annotator](https://bioportal.bioontology.org/annotator) - Syntactic ontology matching
- [text2term](https://github.com/ccb-hms/text2term) - Term mapping toolkit

## License

MIT
