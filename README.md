
# Whisper and RAG Project

This repository automates the transcription, summarization, and retrieval of audio data using OpenAI Whisper for transcription and a Retrieval-Augmented Generation (RAG) pipeline for querying past summaries.

## Features
- **Audio Transcription**: Transcribes audio files into text using Whisper.
- **Text Preprocessing**: Cleans and segments the transcriptions for structured summarization.
- **Summarization**: Summarizes transcripts using a custom LLM (e.g., Ollama's models).
- **RAG Pipeline**: Enables querying of previously generated summaries to extract insights.

---

## Setup and Installation

### Prerequisites
1. **Python**: Ensure Python 3.10 or later is installed.
2. **FFmpeg**: Required for audio file processing.
   - **Ubuntu/Debian**:
     ```bash
     sudo apt update
     sudo apt install ffmpeg
     ```
   - **MacOS (Homebrew)**:
     ```bash
     brew install ffmpeg
     ```
   - **Windows**:
     Download and add FFmpeg to your PATH: [FFmpeg Downloads](https://ffmpeg.org/download.html).

3. **Git**: Ensure Git is installed for cloning the repository.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/bysocode/MeetingSummurizer
   cd MeetingSummurizer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Use the Project

### 1. **Audio Transcription**
Place your audio file (e.g., `2minssample.mp3`) in the project directory and run the transcription process:
```bash
python app.py
```

The application will:
1. Transcribe the audio file.
2. Clean and preprocess the transcript.
3. Generate a summarized version of the transcript using an LLM.
4. Save the summaries in the `summaries_txt` folder.

### 2. **RAG Pipeline**
The project supports querying past summaries using the RAG pipeline:
```bash
python app.py
```
This command will also:
- Build an index using summaries in the `summaries_txt` directory.
- Allow you to query the index for insights.

### 3. **Fine-Tuning Models**
To fine-tune models with your RAG documents:
1. Prepare your dataset in JSON format from the RAG directory.
2. Use the `rag_pipeline.py` to preprocess the data for training.
3. Fine-tune models using Hugging Face's Trainer API or similar frameworks.

---

## File Structure
- `whisperapp.py`: Handles transcription using Whisper.
- `preprocess_transcript.py`: Cleans and segments transcripts for further processing.
- `ollamaInterface.py`: Interacts with the custom LLM for summarization.
- `rag_pipeline.py`: Implements the RAG pipeline for document indexing and querying.
- `app.py`: End-to-end script that ties all components together.
- `summaries_txt/`: Directory for storing generated summaries.

---

## Example Output
**Input**: `2minssample.mp3`

**Transcript** (Partial):
```
This is the transcript of the meeting. It's the kickoff meeting for the project. The primary focus is on understanding...
```

**Summary**:
```
The meeting focused on project kick-off, with discussions about timelines, responsibilities, and deliverables. Key points...
```

**Querying the RAG Pipeline**:
```bash
What are the key takeaways from past meetings?
```

**RAG Response**:
```
Key takeaways:
1. Project kick-off focused on timelines and responsibilities.
2. Discussions about future deliverables and milestone tracking.
```

