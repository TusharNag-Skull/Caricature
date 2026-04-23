import os
import logging
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Client replaces the old genai.configure() + GenerativeModel() pattern
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-2.5-flash"

# ---------- Shared safety / gen-config ---------- #
SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",       threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",      threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]

GEN_CFG = types.GenerateContentConfig(
    temperature=0.7,
    top_p=0.95,
    top_k=32,
    max_output_tokens=20000,
    safety_settings=SAFETY_SETTINGS,
)


# ---------- Streaming helper ---------- #
def stream_project(prompt: str):
    """
    Streams response chunks from Gemini.
    Each yielded chunk has a .text attribute — same contract as before.
    """
    response = client.models.generate_content_stream(
        model=MODEL,
        contents=prompt,
        config=GEN_CFG,
    )
    for chunk in response:
        if getattr(chunk, "text", None):
            yield chunk


# ---------- Classifier ---------- #
def quick_classify(prompt: str) -> str:
    """
    Returns either 'react' or 'node' based on the prompt content.
    """
    classification_prompt = (
        "Return either node or react based on what do you think this project should be. "
        "Only return a single word either 'node' or 'react'. Do not return anything extra.\n\n"
        f"{prompt}"
    )

    classify_cfg = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=20,
        safety_settings=SAFETY_SETTINGS,
    )

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=classification_prompt,
            config=classify_cfg,
        )
        return response.text.strip().lower()

    except Exception as e:
        logger.error(f"[quick_classify] Error: {e}")
        raise


# ---------- Chat ---------- #
def chat(messages: list[dict], system_prompt: str) -> str:
    """
    Calls Gemini with a system prompt + user messages.

    NOTE: The new SDK supports system_instruction natively via GenerateContentConfig,
    so we no longer need to manually prepend it to the prompt string.
    """
    try:
        # Build contents list from user messages only
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=m["content"])]
            )
            for m in messages
            if m["role"] == "user"
        ]

        chat_cfg = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=32,
            max_output_tokens=20000,
            safety_settings=SAFETY_SETTINGS,
            system_instruction=system_prompt,  # First-class support now — no manual prepending needed
        )

        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=chat_cfg,
        )

        return response.text

    except Exception as e:
        logger.error(f"[chat] Error: {e}")
        raise