"""BC5CDR benchmark loader and evaluator.

The BC5CDR corpus consists of 1500 PubMed abstracts with:
- Chemical and Disease entity annotations (with MESH IDs)
- Chemical-Induced-Disease (CID) relationship annotations

Format (PubTator):
- PMID|t|Title
- PMID|a|Abstract
- PMID\tstart\tend\ttext\ttype\tMESH_ID (entities)
- PMID\tCID\tChemical_ID\tDisease_ID (relations)

Reference: BioCreative V CDR Task
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Entity:
    """An entity annotation from BC5CDR."""
    start: int
    end: int
    text: str
    entity_type: str  # "Chemical" or "Disease"
    mesh_id: str


@dataclass
class Relation:
    """A CID relation from BC5CDR."""
    chemical_id: str
    disease_id: str


@dataclass
class Document:
    """A BC5CDR document with annotations."""
    pmid: str
    title: str
    abstract: str
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Full document text (title + abstract)."""
        return f"{self.title} {self.abstract}"


def load_pubtator(filepath: Path) -> list[Document]:
    """Load documents from a PubTator format file.

    Args:
        filepath: Path to the PubTator file

    Returns:
        List of Document objects with entities and relations
    """
    documents: dict[str, Document] = {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Title line: PMID|t|Title
            if "|t|" in line:
                pmid, _, title = line.partition("|t|")
                if pmid not in documents:
                    documents[pmid] = Document(pmid=pmid, title=title, abstract="")
                else:
                    documents[pmid].title = title

            # Abstract line: PMID|a|Abstract
            elif "|a|" in line:
                pmid, _, abstract = line.partition("|a|")
                if pmid not in documents:
                    documents[pmid] = Document(pmid=pmid, title="", abstract=abstract)
                else:
                    documents[pmid].abstract = abstract

            # Tab-separated: entity or relation
            elif "\t" in line:
                parts = line.split("\t")
                pmid = parts[0]

                if pmid not in documents:
                    documents[pmid] = Document(pmid=pmid, title="", abstract="")

                # CID relation: PMID\tCID\tChemical_ID\tDisease_ID
                if len(parts) == 4 and parts[1] == "CID":
                    documents[pmid].relations.append(
                        Relation(chemical_id=parts[2], disease_id=parts[3])
                    )
                # Entity: PMID\tstart\tend\ttext\ttype\tMESH_ID
                elif len(parts) >= 6:
                    try:
                        documents[pmid].entities.append(
                            Entity(
                                start=int(parts[1]),
                                end=int(parts[2]),
                                text=parts[3],
                                entity_type=parts[4],
                                mesh_id=parts[5],
                            )
                        )
                    except ValueError:
                        # Skip malformed entity lines
                        pass

    return list(documents.values())


def evaluate_relations(
    predicted: list[tuple[str, str]],
    gold: list[Relation],
) -> dict[str, float]:
    """Evaluate predicted relations against gold annotations.

    Args:
        predicted: List of (chemical_id, disease_id) tuples
        gold: List of Relation objects from the gold annotations

    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    predicted_set = set(predicted)
    gold_set = {(r.chemical_id, r.disease_id) for r in gold}

    if not predicted_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    true_positives = len(predicted_set & gold_set)

    precision = true_positives / len(predicted_set) if predicted_set else 0.0
    recall = true_positives / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def get_corpus_path(split: str = "test") -> Path:
    """Get the path to a BC5CDR corpus split.

    Args:
        split: One of "train", "dev", or "test"

    Returns:
        Path to the PubTator file
    """
    base = Path(__file__).parent / "data" / "CDR_Data" / "CDR.Corpus.v010516"

    split_map = {
        "train": "CDR_TrainingSet.PubTator.txt",
        "dev": "CDR_DevelopmentSet.PubTator.txt",
        "test": "CDR_TestSet.PubTator.txt",
    }

    if split not in split_map:
        raise ValueError(f"Unknown split: {split}. Must be one of {list(split_map.keys())}")

    return base / split_map[split]


if __name__ == "__main__":
    # Quick test of loader
    test_path = get_corpus_path("test")
    docs = load_pubtator(test_path)
    print(f"Loaded {len(docs)} documents")

    total_entities = sum(len(d.entities) for d in docs)
    total_relations = sum(len(d.relations) for d in docs)
    print(f"Total entities: {total_entities}")
    print(f"Total relations: {total_relations}")

    # Show first document
    if docs:
        d = docs[0]
        print(f"\nFirst document (PMID: {d.pmid}):")
        print(f"  Title: {d.title[:80]}...")
        print(f"  Entities: {len(d.entities)}")
        print(f"  Relations: {len(d.relations)}")
        for r in d.relations:
            print(f"    CID: {r.chemical_id} -> {r.disease_id}")
