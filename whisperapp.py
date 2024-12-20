import ssl
from whisper import load_model

# Disable SSL verification (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Load Whisper model globally
print("Loading Whisper model...")
whisper_model = load_model("base")


def transcribe_audio(audio_file: str, language: str = "en", task: str = "transcribe") -> list:
    """
    Transcribe an audio file using Whisper and return the transcription as a list of segments.

    Args:
        audio_file (str): Path to the audio file to transcribe.
        language (str): Language of the audio file. Default is 'en'.
        task (str): Whisper task (e.g., 'transcribe' or 'translate'). Default is 'transcribe'.

    Returns:
        list: Transcription segments with start and end times.
    """
    print(f"Transcribing audio file: {audio_file} with Whisper...")
    result = whisper_model.transcribe(audio_file, language=language, task=task)
    print("Transcription completed!")
    return result["segments"]
