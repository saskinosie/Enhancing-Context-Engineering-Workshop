from pydantic import BaseModel
from pydantic_ai import Agent


# Each intent has a set of required and optional slots
SLOT_DEFINITIONS = {
    "billing": {
        "required": ["price_keyword"],
        "optional": ["department", "product_type"],
        "descriptions": {
            "price_keyword": "Keywords related to price or budget (e.g. 'affordable', 'under $50', 'luxury', 'sale')",
            "department": "The department or gender category (e.g. 'Menswear', 'Ladieswear', 'Sport')",
            "product_type": "The type of product (e.g. 'Dress', 'Jacket', 'Trousers')",
        },
    },
    "product": {
        "required": ["product_type"],
        "optional": ["department", "feature_keyword"],
        "descriptions": {
            "product_type": "The type of product to search for (e.g. 'Dress', 'Jacket', 'T-shirt', 'Trousers')",
            "department": "The department or gender category (e.g. 'Menswear', 'Ladieswear', 'Sport')",
            "feature_keyword": "Specific style, color, or feature to look for (e.g. 'floral', 'waterproof', 'formal')",
        },
    },
    "support": {
        "required": ["issue_keyword"],
        "optional": ["department", "product_type"],
        "descriptions": {
            "issue_keyword": "What the user needs help finding (e.g. 'warm winter coat', 'formal event outfit', 'waterproof hiking jacket')",
            "department": "The department or gender category (e.g. 'Menswear', 'Ladieswear', 'Sport')",
            "product_type": "The type of product (e.g. 'Jacket', 'Dress', 'Shoes')",
        },
    },
}


class SlotExtractionResult(BaseModel):
    extracted_slots: dict[str, str]
    missing_required: list[str]
    clarifying_question: str | None


slot_extractor = Agent(
    model="openai:gpt-4.1-mini",
    output_type=SlotExtractionResult,
    instructions="""
    You extract slot values from a user message for a given intent.

    You will receive:
    1. The user's message
    2. The current intent
    3. The slot definitions (required and optional slots with descriptions)
    4. Any slots already filled from previous turns

    Your job:
    - Extract any slot values present in the user's message
    - Identify which required slots are still missing
    - If required slots are missing, generate a natural clarifying question
    - NEVER hallucinate slot values — only extract what the user actually said
    - If a slot value is ambiguous, ask for clarification rather than guessing

    Return extracted_slots as a dict of slot_name: value pairs.
    Return missing_required as a list of slot names that are required but not yet filled.
    Return clarifying_question as a natural question to ask, or null if all required slots are filled.
    """,
)


async def extract_slots(
    utterance: str,
    intent: str,
    existing_slots: dict[str, str] | None = None,
) -> SlotExtractionResult:
    if existing_slots is None:
        existing_slots = {}

    slot_def = SLOT_DEFINITIONS.get(intent, {})

    prompt = f"""
    User message: "{utterance}"
    Intent: {intent}
    Slot definitions: {slot_def}
    Already filled slots: {existing_slots}

    Extract slot values from the user message and identify missing required slots.
    """

    result = await slot_extractor.run(prompt)
    return result.output
