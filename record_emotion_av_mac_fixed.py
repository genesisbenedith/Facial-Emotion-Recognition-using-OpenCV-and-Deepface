import cv2
import sounddevice as sd
import soundfile as sf
import numpy as np
import subprocess
from deepface import DeepFace
import threading

# ------------------------
# Audio Recording Settings
# ------------------------
AUDIO_FILENAME = "audio_temp.wav"
AUDIO_RATE = 44100
AUDIO_CHANNELS = 1

audio_frames = []
audio_stream = None

def audio_callback(indata, frames, time, status):
    if status:
        print("Audio status:", status)
    audio_frames.append(indata.copy())

def start_audio_stream():
    global audio_stream
    audio_stream = sd.InputStream(
        samplerate=AUDIO_RATE,
        channels=AUDIO_CHANNELS,
        dtype="float32",
        blocksize=1024,
        callback=audio_callback
    )
    audio_stream.start()
    print("Audio stream started...")

def stop_audio_stream():
    global audio_stream
    if audio_stream:
        audio_stream.stop()
        audio_stream.close()
        audio_stream = None
        print("Audio stream stopped.")

# ------------------------
# Video Recording Settings
# ------------------------
VIDEO_FILENAME = "video_temp.mp4"
FINAL_FILENAME = "emotion_with_audio_mac_fixed.mp4"

fps = 20.0
frame_width = 640
frame_height = 480

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter(VIDEO_FILENAME, fourcc, fps, (frame_width, frame_height))

# Emotion detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ------------------------
# Start Audio
# ------------------------
start_audio_stream()

# ------------------------
# Start Webcam
# ------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

print("Recording video + audio... Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Emotion detection
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        face_roi = rgb[y:y+h, x:x+w]
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Save video frame
    video_out.write(frame)

    # Display
    cv2.imshow("Recording Emotion + Audio (macOS)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------
# Stop All Streams
# ------------------------
cap.release()
video_out.release()
cv2.destroyAllWindows()

stop_audio_stream()

# ------------------------
# Save Recorded Audio
# ------------------------
print("Saving audio...")

audio_data = np.concatenate(audio_frames, axis=0)
sf.write(AUDIO_FILENAME, audio_data, AUDIO_RATE)

print("Audio saved.")

# ------------------------
# Merge Audio + Video with FFmpeg
# ------------------------
print("Merging audio + video...")

subprocess.call([
    "ffmpeg",
    "-y",
    "-i", VIDEO_FILENAME,
    "-i", AUDIO_FILENAME,
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-c:v", "copy",
    "-c:a", "aac",
    "-b:a", "192k",
    "-shortest",
    FINAL_FILENAME
])

print(f"Done! Final file saved as: {FINAL_FILENAME}")
