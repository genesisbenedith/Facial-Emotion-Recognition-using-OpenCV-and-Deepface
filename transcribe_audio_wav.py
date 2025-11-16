from openai import OpenAI

import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

audio_file = open("audio_temp.wav", "rb")

transcript = client.audio.transcriptions.create(
    model="gpt-4o-transcribe",  # The updated Whisper replacement
    file=audio_file
)

print(transcript.text)