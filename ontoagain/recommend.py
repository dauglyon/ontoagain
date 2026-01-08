"""RECOMMEND agent - suggest ontologies for a paper based on extracted concepts."""

from pathlib import Path

from ontoagain.identify import DEFAULT_MODEL, identify, parse_xml_output
from ontoagain.llm import call_llm, get_client
from ontoagain.models import Concept, OntologyRecommendation
from ontoagain.xml_utils import parse_xml_fragment, safe_get_text

# Load prompt template
PROMPT_PATH = Path(__file__).parent / "prompts" / "recommend.txt"


def load_prompt() -> str:
    """Load the recommend prompt template."""
    return PROMPT_PATH.read_text()


def format_concepts_for_prompt(concepts: list[Concept]) -> str:
    """Format extracted concepts for the recommendation prompt."""
    lines = []
    for i, concept in enumerate(concepts, 1):
        lines.append(f"{i}. {concept.text} - {concept.context}")
    return "\n".join(lines)


def parse_recommendations(xml_text: str) -> list[OntologyRecommendation]:
    """Parse XML output from recommend LLM call."""
    recommendations = []

    root = parse_xml_fragment(xml_text)

    for onto_elem in root.findall(".//ontology"):
        onto_id = safe_get_text(onto_elem.find("id"))
        name = safe_get_text(onto_elem.find("name"))
        relevance = safe_get_text(onto_elem.find("relevance"))
        examples_text = safe_get_text(onto_elem.find("example_concepts"))
        examples = [e.strip() for e in examples_text.split(",") if e.strip()]
        url = safe_get_text(onto_elem.find("download_url"))

        if onto_id:
            recommendations.append(
                OntologyRecommendation(
                    id=onto_id,
                    name=name,
                    relevance=relevance,
                    example_concepts=examples,
                    download_url=url,
                )
            )

    return recommendations


def recommend(
    text: str,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
    max_chars: int = 30000,
) -> tuple[list[OntologyRecommendation], list[Concept]]:
    """Recommend ontologies for a paper.

    Args:
        text: Plain text of the paper
        model: LLM model to use
        verbose: Print debug info
        max_chars: Maximum characters to process (default 30k for speed)

    Returns:
        Tuple of (recommendations, extracted_concepts)
    """
    # Truncate if needed
    if len(text) > max_chars:
        if verbose:
            print(f"Truncating paper from {len(text)} to {max_chars} chars...")
        text = text[:max_chars]

    # Step 1: Extract concepts
    if verbose:
        print("Step 1: Extracting concepts...")

    results = identify([("input", text)], model=model, verbose=verbose)
    tagged_text = results[0][1] if results else ""
    concepts = parse_xml_output(tagged_text)

    if verbose:
        print(f"Extracted {len(concepts)} concepts")
        for c in concepts[:5]:
            print(f"  - {c.text}: {c.context}")
        if len(concepts) > 5:
            print(f"  ... and {len(concepts) - 5} more")
        print()

    if not concepts:
        if verbose:
            print("No concepts extracted - cannot recommend ontologies")
        return [], []

    # Step 2: Recommend ontologies
    if verbose:
        print("Step 2: Analyzing concepts to recommend ontologies...")

    prompt_template = load_prompt()
    concepts_text = format_concepts_for_prompt(concepts)
    prompt = prompt_template.replace("{concepts}", concepts_text)

    client, model_name = get_client(model)
    messages = [{"role": "user", "content": prompt}]
    result = call_llm(client, model_name, messages)

    if verbose:
        print(f"LLM response received ({len(result)} chars)")
        print()

    recommendations = parse_recommendations(result)

    if verbose:
        print(f"Parsed {len(recommendations)} ontology recommendations")

    return recommendations, concepts
