import json
from whisperapp import transcribe_audio
from preprocess_transcript import clean_transcript, tokenize_and_chunk, format_for_prompt
from ollamaInterface import CustomOllama
from rag_pipeline import build_rag_index, query_rag_index
import os

# Input file
audio_file = "2minssample.mp3"

# Run transcription
print("Running transcription...")
transcription_segments = transcribe_audio(audio_file)

# Combine transcription segments into a single string
full_transcript = " ".join([segment['text'] for segment in transcription_segments])

print("\n=== Raw Transcription ===")
print(full_transcript)

# Preprocess the transcript
print("\nCleaning the transcript...")
cleaned_transcript = clean_transcript(full_transcript)

print("\n=== Cleaned Transcript ===")
print(cleaned_transcript)

# Tokenize and chunk the cleaned transcript
print("\nTokenizing and chunking the transcript...")
chunks = tokenize_and_chunk(cleaned_transcript, max_tokens=500)

print("\n=== Transcript Chunks ===")
for idx, chunk in enumerate(chunks):
    print(f"Chunk {idx + 1}:\n{chunk}\n")

# Format for LLM prompt
print("\nFormatting transcript chunks for LLM prompt...")
formatted_prompt = format_for_prompt(chunks)

print("\n=== Formatted Prompt ===")
print(formatted_prompt)

# Save the formatted prompt to a file
output_file = "formatted_prompt.md"
with open(output_file, "w") as f:
    f.write(formatted_prompt)

print(f"\nFormatted prompt saved to {output_file}")

# Send to LLM for summarization
print("\nSending to LLM for summarization...")
# Define the template for summarization
template = """Transcript:
{question}

Summary: Let's summarize step by step."""

# Create an instance of CustomOllama
ollama_instance = CustomOllama(model_name="llama3.2:latest")

# Use the formatted prompt as the question for summarization
question = formatted_prompt
summarized_text = ollama_instance.send_prompt(template=template, question=question)

print("\n=== Summarized Text ===")
print(summarized_text)

# Save the summarized text to a file
summary_file = "summarized_text.txt"
with open(summary_file, "w") as f:
    f.write(summarized_text)

print(f"\nSummarized text saved to {summary_file}")

# Add summarized text to RAG directory
rag_data_dir = "summaries_txt"
os.makedirs(rag_data_dir, exist_ok=True)

# Save the summary in a unique JSON file
summary_file_path = os.path.join(rag_data_dir, "summary_0.json")
with open(summary_file_path, "w", encoding="utf-8") as f:
    json.dump({"summary": summarized_text}, f)

print(f"\nSummarized text saved for RAG in {summary_file_path}")

# Build the RAG index
print("\nBuilding the RAG index...")
index = build_rag_index(rag_data_dir)

if index:
    # Query the RAG index
    print("\nQuerying the RAG index for insights...")
    query = "What are some detail you could to give to the meeting?"
    response = query_rag_index(index, query)

    print("\n=== RAG Insights ===")
    print(response)
else:
    print("\nNo valid RAG index could be built.")
