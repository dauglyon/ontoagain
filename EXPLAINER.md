# OntoAgain Technical Explainer

OntoAgain is an LLM-powered system for extracting scientific concepts from research papers and mapping them to ontology terms. This document explains the architecture, compares it to OntoGPT, presents performance benchmarks, and discusses future work.

## Overview

OntoAgain processes scientific text in a two-phase pipeline:

```
┌─────────────────┐     ┌─────────────────────┐     ┌───────────────────┐
│  Research Paper │────▶│  IDENTIFY (LLM)     │────▶│  Concepts (XML)   │
│     (.txt)      │     │  Extract concepts   │     │  <concept>...</>  │
└─────────────────┘     └─────────────────────┘     └────────┬──────────┘
                                                             │
                                                             ▼
                        ┌─────────────────────┐     ┌───────────────────┐
                        │  DISAMBIGUATE       │────▶│  Tagged Output    │
                        │  Vector search +    │     │  <concept>        │
                        │  LLM verification   │     │    <match .../>   │
                        └─────────────────────┘     └───────────────────┘
```

## How It Works

### Phase 1: IDENTIFY

The IDENTIFY phase uses an LLM (default: Claude Sonnet) to extract scientific concepts from text. The LLM:

1. Reads the paper text with a structured prompt
2. Identifies scientific entities (genes, proteins, chemicals, species, processes, etc.)
3. Wraps each concept in an XML tag with metadata:

```xml
<concept context="programmed cell death" search="apoptosis; programmed cell death">apoptosis</concept>
```

The tag includes:
- **text** (content): The surface form as it appears in the paper
- **context**: Resolved meaning with expanded abbreviations
- **search**: Terms optimized for ontology matching

**Chunking for Long Documents**: Papers exceeding 20K characters are automatically split into chunks at paragraph boundaries with 500-character overlap. Each chunk is processed independently and outputs are merged.

**Ontology Grounding**: When an index is available, ontology metadata (descriptions and term formats) is injected into the prompt. This helps the LLM generate search terms that match the ontology vocabulary.

### Phase 2: DISAMBIGUATE

The DISAMBIGUATE phase maps extracted concepts to ontology terms:

1. **Vector Search**: For each concept, the search terms are embedded using BGE-M3 and queried against a LanceDB vector index containing 3.3M+ ontology terms
2. **Candidate Retrieval**: Top-k candidates (default: 20) are retrieved for each concept
3. **Batching by Overlap**: Concepts with overlapping candidate sets are grouped into batches to reduce LLM calls
4. **LLM Verification**: Each batch is sent to the LLM with candidates, which selects the best matching term(s)

The output adds nested `<match/>` elements with ontology mappings:

```xml
<concept context="programmed cell death" search="apoptosis; programmed cell death">apoptosis
  <match ontology="GO" id="GO:0006915" label="apoptotic process"/>
</concept>
```

### Index Architecture

OntoAgain uses LanceDB with IVF-PQ indexing for fast approximate nearest neighbor search:

- **Embedding Model**: BGE-M3 (BAAI/bge-m3) - 1024-dimensional embeddings
- **Index Type**: IVF-PQ with partitions proportional to dataset size
- **GPU Acceleration**: Supported for both indexing and search
- **Embedding Text**: Combines term label, synonyms, and definition for richer semantic matching

```python
# Example: embedding text construction
parts = [term.label] + term.synonyms + [term.definition]
text = " | ".join(parts)
```

## Comparison with OntoGPT

| Aspect | OntoAgain | OntoGPT |
|--------|-----------|---------|
| **Approach** | Two-phase pipeline (extract → match) | Schema-driven extraction (SPIRES) |
| **Schema** | No schema required | Requires LinkML schema templates |
| **Output** | XML with inline concept tags | JSON/YAML/RDF structured instances |
| **Grounding** | Vector search + LLM verification | Direct LLM grounding with ontology context |
| **Training** | Zero-shot (no training data) | Zero-shot (no training data) |
| **Customization** | Ontology config file with descriptions | Pre-built or custom LinkML templates |

### Key Differences

**1. Schema vs Schema-free**

OntoGPT requires a LinkML schema defining the expected structure of extracted information. This makes it powerful for structured relation extraction but requires schema authoring.

OntoAgain is schema-free - it extracts all scientific concepts without predefined structure. This is more flexible for exploratory annotation but doesn't capture relations between concepts.

**2. Grounding Strategy**

OntoGPT relies on the LLM to directly ground entities to ontology terms based on context in the prompt.

OntoAgain separates concerns: vector search retrieves candidates efficiently, then LLM verifies the best match. This allows scaling to very large ontologies (3M+ terms) that wouldn't fit in context.

**3. Output Format**

OntoGPT produces structured outputs matching the schema (instances with typed attributes).

OntoAgain produces inline-annotated text preserving the original document structure with concept boundaries marked.

**4. Use Case Focus**

OntoGPT excels at: structured knowledge extraction, relation extraction, populating knowledge bases.

OntoAgain excels at: document annotation, term tagging, linking papers to ontologies.

## Performance Benchmarks

Tested on a Nature Genetics paper (109K characters) about DNA methylation in eukaryotes.

### Pipeline Timing (5K character excerpt, 120 concepts)

| Phase | Time | Concepts/sec |
|-------|------|--------------|
| IDENTIFY | 82s | 1.5 |
| DISAMBIGUATE | 66s | 1.8 |
| **Total** | 148s | 0.8 |

### Index Statistics

| Ontology | Terms |
|----------|-------|
| GO (Gene Ontology) | 39,365 |
| CHEBI (Chemical Entities) | 204,727 |
| NCBITaxon (Taxonomy) | 2,708,808 |
| PR (Protein Ontology) | 360,304 |
| **Total** | 3,313,204 |

### Match Rate

- 120 concepts extracted from 5K excerpt
- 71 matched to ontology terms (59%)
- 21 LLM batches (average 5.7 concepts/batch)

### Full Paper Processing (109K characters)

- 6 chunks (20K chars each with 500 char overlap)
- 1,622 total concepts extracted
- ~260K characters output XML

## Future Work

### 1. Better Embeddings

**Current**: BGE-M3 (general-purpose multilingual embeddings)

**Improvements**:
- Domain-specific embeddings trained on biomedical text (e.g., PubMedBERT, BioLord)
- Multi-vector representations that capture synonyms separately
- Ensemble approaches combining multiple embedding models
- Fine-tuning on ontology-specific data

### 2. Parallelization

**Current Bottlenecks**:
- Sequential LLM calls during IDENTIFY chunking
- Sequential batches during DISAMBIGUATE
- CPU-only vector search during inference

**Improvements**:
- Parallel chunk processing with asyncio for IDENTIFY
- Concurrent batch processing for DISAMBIGUATE (with rate limiting)
- GPU-accelerated search for large-scale batch queries
- Streaming processing for very large documents

### 3. Smaller Models for IDENTIFY

**Current**: Claude Sonnet for both phases (~$0.003/1K input, $0.015/1K output)

**Improvements**:
- Use smaller/cheaper models for IDENTIFY (Claude Haiku, GPT-4o-mini)
- Reserve Sonnet/Opus for DISAMBIGUATE where precision matters
- Local models (Llama 3, Mistral) for cost-sensitive applications
- Fine-tuned small models specifically for concept extraction

**Tradeoff Analysis**:
| Model | Speed | Cost | Quality |
|-------|-------|------|---------|
| Claude Haiku | Fast | Low | Good for clear concepts |
| Claude Sonnet | Medium | Medium | Best for ambiguous cases |
| Local Llama 3 | Varies | Free | Quality varies by domain |

### 4. Better Context Engineering

**Current Limitations**:
- Abbreviations defined early may be forgotten in later chunks
- Cross-references between sections are lost
- No document-level reasoning

**Improvements**:
- Two-pass processing: extract abbreviations/context first, then annotate
- Sliding window with larger overlap for context continuity
- Document summarization injected into each chunk prompt
- Coreference resolution pre-processing
- Section-aware chunking (intro, methods, results, discussion)

### 5. Additional Enhancements

- **Confidence Scores**: Add similarity scores from vector search to output
- **Negative Selection**: Better prompts for "no match" cases
- **Multi-ontology Ranking**: Prioritize ontologies by relevance to paper topic
- **Interactive Mode**: Human-in-the-loop verification for ambiguous matches
- **Batch API**: Process multiple papers concurrently
- **Incremental Indexing**: Add new ontologies without full rebuild

## Architecture Reference

```
ontoagain/
├── cli.py           # Typer CLI (identify, disambiguate, index, recommend)
├── identify.py      # Concept extraction with chunking
├── disambiguate.py  # Ontology matching with batching
├── recommend.py     # Ontology recommendations
├── index.py         # LanceDB vector index (build + search)
├── models.py        # Pydantic data models
├── llm.py           # LiteLLM wrapper (multi-provider)
├── xml_utils.py     # XML parsing with error handling
└── prompts/         # LLM prompt templates
    ├── identify.txt
    ├── disambiguate_batch.txt
    └── recommend.txt
```

## Dependencies

- **LiteLLM**: Multi-provider LLM abstraction
- **Instructor**: Structured outputs from LLMs
- **LanceDB**: Vector database with IVF-PQ indexing
- **Sentence-Transformers**: BGE-M3 embeddings
- **OAK (ontology-access-kit)**: OBO/OWL ontology parsing
- **Pydantic**: Data validation and models
- **Typer**: CLI framework

## Sources

- [OntoGPT GitHub](https://github.com/monarch-initiative/ontogpt)
- [OntoGPT Documentation](https://monarch-initiative.github.io/ontogpt/)
- [SPIRES Paper (arXiv)](https://arxiv.org/pdf/2304.02711)
