from pydantic import BaseModel
from pydantic_ai import Agent


class IntentResult(BaseModel):
    intent: str
    confidence: float
    reasoning: str


SUPPORTED_INTENTS = ["billing", "product", "support", "unknown"]

intent_classifier = Agent(
    model="openai:gpt-4.1-mini",
    output_type=IntentResult,
    instructions=f"""
    You classify user utterances into one of these intents: {SUPPORTED_INTENTS}

    This is a fashion retail assistant. Intent definitions:
    - billing: Questions about pricing, budget, affordability, discounts, or value.
      e.g. "What's the cheapest option?", "Do you have anything on sale?", "I have a $50 budget"
    - product: Questions about browsing, discovering, or comparing products by category,
      style, color, or department.
      e.g. "What jackets do you have for men?", "Show me dresses", "I'm looking for something to wear"
    - support: Questions asking for detailed help finding something specific, sizing/fit
      guidance, material details, or product specifications.
      e.g. "Help me find a waterproof jacket for hiking", "I need something for a formal event",
      "What's the warmest coat you have?"
    - unknown: The query does not clearly fit any of the above intents.

    Return the most likely intent, a confidence score between 0 and 1,
    and a brief reasoning for your classification.

    Be conservative with confidence — only score above 0.8 if the intent is very clear.
    If the query could reasonably be two intents, pick the stronger one but lower the confidence.
    """,
)


async def classify_intent(utterance: str) -> IntentResult:
    result = await intent_classifier.run(f"Classify this user message: {utterance}")
    return result.output
