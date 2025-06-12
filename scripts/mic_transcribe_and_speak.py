import speech_recognition as sr
import subprocess
import pyttsx3
import os
import tempfile

# Paths - adjust these to your setup
whisper_cli_path = r"C:\Users\Lehto\whisper.cpp\build\bin\Release\whisper-cli.exe"
model_path = r"C:\Users\Lehto\whisper.cpp\models\ggml-base.en.bin"

wake_word = "hey assistant"

engine = pyttsx3.init()
recognizer = sr.Recognizer()
mic = sr.Microphone()

def listen_for_wake_word():
    print("Listening for wake word...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio).lower()
        print(f"Heard: {text}")
        if wake_word in text:
            return True
    except sr.UnknownValueError:
        pass
    except sr.RequestError as e:
        print(f"API error: {e}")
    return False

def record_audio(filename, duration=5):
    print("Recording audio...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.record(source, duration=duration)
    with open(filename, "wb") as f:
        f.write(audio.get_wav_data())
    print(f"Saved recording to {filename}")

def transcribe_audio(audio_file):
    print("Transcribing audio...")
    result = subprocess.run([
        whisper_cli_path,
        "-m", model_path,
        "-f", audio_file,
        "--language", "en"
    ], capture_output=True, text=True)
    # Extract transcription from output - customize this based on whisper-cli output format
    output = result.stdout.strip()
    print(f"Transcription: {output}")
    return output

def speak(text):
    print(f"Speaking: {text}")
    engine.say(text)
    engine.runAndWait()

def main():
    while True:
        if listen_for_wake_word():
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                record_audio(temp_audio.name)
                transcription = transcribe_audio(temp_audio.name)
                speak(transcription)
                os.unlink(temp_audio.name)

if __name__ == "__main__":
    main()
