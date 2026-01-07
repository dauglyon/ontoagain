"""Data models for OntoAgain pipeline."""

from pydantic import BaseModel


class Concept(BaseModel):
    """A scientific concept extracted from text."""

    text: str  # Surface text from paper
    context: str  # Expanded form, synonyms, ontology terms (semicolon-separated)
    start: int  # Start position in original text
    end: int  # End position in original text


class OntologyMatch(BaseModel):
    """A match to an ontology term."""

    ontology: str  # e.g., "GO", "ChEBI"
    id: str  # e.g., "GO:0006915"
    label: str  # e.g., "apoptotic process"


class TaggedConcept(BaseModel):
    """A concept with its ontology mappings."""

    text: str
    context: str
    start: int
    end: int
    matches: list[OntologyMatch]


class PipelineResult(BaseModel):
    """Complete result from the tagging pipeline."""

    concepts: list[TaggedConcept]  # Concepts with ontology mappings
    stats: dict  # matched/unmatched counts


class OntologyTerm(BaseModel):
    """An ontology term for indexing."""

    id: str
    ontology: str
    label: str
    synonyms: list[str]
    definition: str


class OntologyRecommendation(BaseModel):
    """A recommended ontology for a paper."""

    id: str
    name: str
    relevance: str
    example_concepts: list[str]
    download_url: str


class OntologyConfig(BaseModel):
    """Configuration for a single ontology."""

    path: str  # Path to OBO/OWL file
    description: str  # What the ontology contains
    term_format: str  # How terms are formatted/phrased


class OntologiesConfig(BaseModel):
    """Configuration file for ontology indexing."""

    ontologies: list[OntologyConfig]


class OntologyMetadata(BaseModel):
    """Stored metadata about an indexed ontology."""

    name: str  # Short name (e.g., "GO", "CHEBI")
    description: str  # What the ontology contains
    term_format: str  # How terms are formatted
    term_count: int  # Number of terms indexed


class RawRelationship(BaseModel):
    """Relationship extracted by LLM (before predicate disambiguation)."""

    subject_id: str  # Ontology ID from <M/> tag (e.g., "D003000")
    object_id: str  # Ontology ID from <M/> tag (e.g., "D006973")
    predicate: str  # Relationship type only: "induces", "inhibits", "regulates"
    concept_ids: list[str]  # Concept IDs (n values) supporting this relationship
    summary: str  # Explanation of the relationship in document context


class Relationship(BaseModel):
    """Fully resolved relationship (after predicate disambiguation)."""

    subject_id: str  # Ontology ID (e.g., "D003000")
    object_id: str  # Ontology ID (e.g., "D006973")
    predicate_id: str  # Relationship ontology ID (e.g., "RO:0002436")
    predicate_label: str  # Relationship label (e.g., "causally_upstream_of")
    predicate_raw: str  # Original LLM predicate (e.g., "induces")
    concept_ids: list[str]  # Concept IDs (n values) supporting this relationship
    summary: str  # Explanation of the relationship in document context
