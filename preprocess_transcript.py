import re
from transformers import GPT2Tokenizer


def clean_transcript(transcript: str) -> str:
    """
    Clean the transcript by removing filler words, redundant punctuation, and normalizing spaces.

    Args:
        transcript (str): The raw transcript.

    Returns:
        str: Cleaned transcript.
    """
    # List of filler words and phrases
    fillers = [
        r"\b(um|uh|er|ah|hmm|uh-huh|mm-hmm)\b",
        r"\b(you know|I mean|like|sort of|kind of|basically|actually|literally|just|well|so|right|okay|alright|I guess|maybe|probably|somehow|I think|sorta|kinda)\b",
        r"\b(if you don’t mind|you see|please|excuse me|sorry|innit|erm|yeah-nah|argh|you know what I mean|to be honest|at the end of the day)\b"
    ]

    # Create a regex pattern for fillers
    filler_pattern = "|".join(fillers)

    # Remove filler words and phrases (case insensitive)
    cleaned = re.sub(filler_pattern, "", transcript, flags=re.IGNORECASE)

    # Normalize spaces and punctuation
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()  # Reduce multiple spaces to one
    cleaned = re.sub(r"([.?!])\s+", r"\1 ", cleaned)  # Ensure proper spacing after punctuation
    cleaned = re.sub(r"([?!]){2,}", r"\1", cleaned)   # Limit repeated punctuation

    # Sentence capitalization
    sentences = re.split(r"(?<=[.?!])\s+", cleaned)  # Split into sentences
    sentences = [s.capitalize() for s in sentences]  # Capitalize each sentence
    return " ".join(sentences)


def tokenize_and_chunk(transcript: str, max_tokens: int = 500, overlap: int = 50, tokenizer=None) -> list:
    """
    Tokenize and split the transcript into manageable chunks.

    Args:
        transcript (str): The cleaned transcript.
        max_tokens (int): Maximum number of tokens per chunk.
        overlap (int): Number of overlapping tokens between chunks.
        tokenizer: Optional GPT-2 tokenizer instance.

    Returns:
        list: List of transcript chunks.
    """
    # Validate input
    if not transcript.strip():
        print("Warning: Empty or invalid transcript provided.")
        return []

    # Load GPT-2 tokenizer if not provided
    tokenizer = tokenizer or GPT2Tokenizer.from_pretrained("gpt2")

    # Tokenize the transcript
    tokens = tokenizer.encode(transcript)

    # Split into chunks with overlap
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens - overlap)]

    # Decode chunks back into text
    return [tokenizer.decode(chunk) for chunk in chunks]


def format_for_prompt(chunks: list, heading: str = "Meeting Transcript") -> str:
    """
    Format the transcript chunks into a structured LLM prompt.

    Args:
        chunks (list): List of transcript chunks.
        heading (str): Heading for the prompt.

    Returns:
        str: Structured prompt.
    """
    formatted = f"# {heading}\n\n"
    for idx, chunk in enumerate(chunks):
        formatted += f"## Chunk {idx + 1}:\n{chunk}\n\n"
    return formatted
