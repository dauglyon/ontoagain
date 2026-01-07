# OntoAgain

LLM-powered ontology term identification, disambiguation, and relationship extraction for scientific papers.

## What It Does

OntoAgain extracts scientific concepts from research papers, maps them to ontology terms, and extracts relationships between them.

**Input** (plain text from a paper):
```
DNA methylation of 6mA in ciliates is associated with transcriptional activation.
```

**Output** (after CONCEPT-EXTRACT → CONCEPT-DISAMBIGUATE → RELATION-EXTRACT → RELATION-DISAMBIGUATE pipeline):

Concepts with ontology mappings:
```xml
<C n="1" q="DNA methylation; epigenetic modification">DNA methylation<M id="GO:0006306" o="GO" l="DNA methylation"/></C> of
<C n="2" q="6mA; N6-methyladenine">6mA<M id="CHEBI:21891" o="CHEBI" l="N6-methyladenine"/></C> in
<C n="3" q="ciliates; Ciliophora">ciliates<M id="NCBITaxon:5878" o="NCBITAXON" l="Ciliophora"/></C> is associated with
<C n="4" q="transcriptional activation">transcriptional activation<M id="GO:0045893" o="GO" l="positive regulation of transcription"/></C>.
```

Extracted relationships (with predicate mapped to RO):
```xml
<relations>
<R s="GO:0006306" o="GO:0045893" p="RO:0002411" pl="causally upstream of" pr="associated_with" e="1,4">DNA methylation is associated with transcriptional activation in ciliates</R>
</relations>
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
│  Research Paper │────▶│  CONCEPT-EXTRACT    │────▶│  concepts.xml     │
│                 │     │  (LLM)              │     │  <C n="1" q=".."> │
└─────────────────┘     └─────────────────────┘     └────────┬──────────┘
                                                             │
                        ┌─────────────────────┐              ▼
                        │  Concept Index      │     ┌───────────────────┐
                        │  (LanceDB + BGE-M3) │────▶│ CONCEPT-DISAMBIG  │
                        │  3M+ terms          │     │ (Vector + LLM)    │
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
                        │  <R s=".." p="..">│
                        └────────┬──────────┘
                                 │
┌─────────────────────┐          ▼
│  Relation Index     │  ┌───────────────────┐
│  (e.g., RO)         │─▶│ RELATION-DISAMBIG │
└─────────────────────┘  │ (Vector + LLM)    │
                         └────────┬──────────┘
                                  │
                                  ▼
                         ┌────────────────────┐
                         │  relations-final.xml│
                         │  <R p="RO:001" ..> │
                         └────────────────────┘
```

## Key Innovations

### 1. Separation of Extraction and Grounding

Unlike OntoGPT which does both in a single pass, OntoAgain separates:

- **CONCEPT-EXTRACT**: LLM extracts concepts with rich context (expanded abbreviations, synonyms, ontology-friendly terms)
- **CONCEPT-DISAMBIGUATE**: Vector search retrieves candidates, LLM verifies best match
- **RELATION-EXTRACT**: LLM finds relationships between matched concepts
- **RELATION-DISAMBIGUATE**: Maps predicate verbs to Relation Ontology terms

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

### 6. Concept ID Tracking

Each concept gets an incrementing ID (`n="1"`, `n="2"`, etc.) that flows through the pipeline. Relationships reference these IDs in the `e` (evidence) attribute, providing traceability from relationships back to the supporting concepts.

### 7. Compact XML Format

Efficient format for pipeline output:
- `<C n="ID" q="context">text<M id="..." o="..." l="..."/></C>` for concepts
- `<R s="subj" o="obj" p="pred" e="1,2">summary</R>` for relationships

## Benchmarks

### BC5CDR: Chemical-Induced-Disease Relation Extraction

Tested on the BioCreative V CDR corpus (Chemical-Disease Relations). The benchmark evaluates the full pipeline: concept extraction, disambiguation to MESH terms, and relationship extraction.

| Metric | OntoAgain |
|--------|-----------|
| **F1 Score** | **60.5%** |
| Precision | 56.1% |
| Recall | 65.7% |

*20 documents from test split, 37 gold relations*

**Index configuration:**
- MESH 2015 (D-numbers + C-numbers): 259,895 terms
- CID relationship ontology for extraction guidance

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

Download ontologies:

```bash
mkdir ontologies
curl -Lo ontologies/go.obo http://purl.obolibrary.org/obo/go.obo
curl -Lo ontologies/chebi.obo http://purl.obolibrary.org/obo/chebi.obo
```

Edit `sample_config.yaml` to point to your ontologies, then build the index:

```bash
onto index sample_config.yaml -O my-index -v
```

The config file contains descriptions that guide the LLM during disambiguation. See `sample_config.yaml` for the format.

### 3. Run the pipeline

```bash
# Step 1: Extract concepts
onto concept-extract paper.txt --index my-index -o concepts.xml -v

# Step 2: Match concepts to ontology terms
onto concept-disambiguate concepts.xml --index my-index -o tagged.xml -v

# Step 3: Extract relationships (optional)
onto relation-extract tagged.xml -o relations.xml -v

# Step 4: Map predicates to RO (optional)
onto relation-disambiguate relations.xml --rel-index ro-index -o relations-mapped.xml -v
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
| `onto benchmark --index <path> --rel-index <path>` | Run BC5CDR benchmark |

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
├── disambiguate.py  # Ontology matching with batching
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
| **Multi-ontology** | Single run across all indexed ontologies | Separate template per ontology |
| **Relationship extraction** | Concept-first with evidence tracking | Template-driven |
| **Use case** | Document annotation | Knowledge base population |

**Key tradeoffs:**
- OntoGPT excels at structured relation extraction with predefined schemas
- OntoAgain excels at exploratory annotation across multiple ontologies without schema design

## Running Benchmarks

### BC5CDR Setup

The BC5CDR benchmark requires:

1. **BC5CDR Corpus** - Download from BioCreative:
   ```bash
   mkdir -p benchmarks/data
   cd benchmarks/data
   # Download CDR_Data.zip from https://biocreative.bioinformatics.udel.edu/resources/corpora/biocreative-v-cdr-corpus/
   unzip CDR_Data.zip
   ```

2. **MESH 2015 Vocabulary** - The corpus uses MESH 2015 annotations:
   ```bash
   cd benchmarks
   # Download from NLM archive
   curl -O https://nlmpubs.nlm.nih.gov/projects/mesh/2015/asciimesh/d2015.bin
   curl -O https://nlmpubs.nlm.nih.gov/projects/mesh/2015/asciimesh/c2015.bin
   cd ..
   ```

3. **Build Indexes**:
   ```bash
   # Build MESH index
   onto index benchmarks/mesh_config.yaml -O indexes/mesh -v

   # Build CID relationship index
   onto index benchmarks/cid_config.yaml -O indexes/cid -v
   ```

4. **Run Benchmark**:
   ```bash
   onto benchmark --index indexes/mesh --rel-index indexes/cid -n 20 -v
   ```

### Sample Config Files

- `sample_config.yaml` - Template for setting up your own ontologies
- `benchmarks/mesh_config.yaml` - MESH vocabulary with C/D-number guidance
- `benchmarks/cid_config.yaml` - Chemical-Induced-Disease relationship extraction

## Future Work

### Additional Benchmarks
- Comparison with OntoGPT on BC5CDR
- Other relation extraction corpora (DDI, ChemProt)

### Better Embeddings
- Domain-specific models (BioLord, PubMedBERT)
- Multi-vector representations for synonyms
- Fine-tuning on ontology structure

### Model Optimization
- Smaller models for extraction (Haiku, local LLMs)
- Reserve larger models for disambiguation
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
