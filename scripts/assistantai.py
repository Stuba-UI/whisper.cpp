import os
import sys
import tempfile
import subprocess
import speech_recognition as sr
import pyttsx3
import time
import pytesseract
import keyboard

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(__file__), relative_path)

pytesseract.pytesseract.tesseract_cmd = resource_path("Tesseract-OCR/tesseract.exe")

from PIL import Image
import mss
import re

from langchain.llms.base import LLM
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

# -------------------- LLM WRAPPER --------------------
class OllamaLLM(LLM):
    def _call(self, prompt: str, stop=None) -> str:
        print("\n--- Prompt sent to Ollama ---")
        print(prompt)
        print("--- End of prompt ---\n")

        cmd = ["ollama", "run", "llama3"]
        result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8")

        print("\n--- Ollama raw output ---")
        print(result.stdout)
        print("--- End of Ollama output ---\n")

        if result.stderr:
            print("--- STDERR ---")
            print(result.stderr)

        return result.stdout.strip()

    @property
    def _identifying_params(self):
        return {"name": "ollama_llama3"}

    @property
    def _llm_type(self):
        return "ollama"

# -------------------- INIT CHAINS --------------------
llm = OllamaLLM()

documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "Ollama lets you run open-source large language models locally.",
    "ConversationalRetrievalChain enables retrieval-augmented generation."
]

embeddings = OllamaEmbeddings(model="llama3")
vectorstore = Chroma.from_texts(documents, embedding=embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever()

# Retrieval QA chain (for PDF/knowledge base)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# Conversational memory chain (for general chat)
chat_history = ChatMessageHistory()

def ask_with_memory(user_input):
    # Build the message list for the prompt
    messages = chat_history.messages.copy()
    # Add the new user message as a HumanMessage object
    from langchain_core.messages import HumanMessage
    messages.append(HumanMessage(content=user_input))
    # Prepare the prompt for the LLM
    prompt_messages = [("system", "You are a helpful assistant.")]
    for msg in messages:
        # Use the class name to determine the role
        role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
        prompt_messages.append((role, msg.content))
    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    response = (prompt | llm).invoke({})
    # Save the new user and AI messages to history
    chat_history.add_user_message(user_input)
    chat_history.add_ai_message(response)
    return response

prompt_template = PromptTemplate.from_template("Question: {question}\nAnswer:")
llm_chain = prompt_template | llm

# -------------------- SETTINGS --------------------
WAKE_WORDS = ["assistant", "hey assistant", "help"]
VISION_PHRASES = [
    "look at my screen",
    "read my screen",
    "what's on my screen",
    "what do you see",
    "what does it say",
    "read the screen",
    "screen content"
]

# -------------------- FUNCTIONS --------------------
last_ocr_text = None

def listen_for_wake_word(recognizer, microphone):
    print("Listening for wake word...")
    with microphone as source:
        audio = recognizer.listen(source, phrase_time_limit=3)
    try:
        text = recognizer.recognize_google(audio).lower()
        print(f"Heard: {text}")
        return any(wake in text for wake in WAKE_WORDS)
    except (sr.UnknownValueError, sr.RequestError):
        return False

def record_audio(recognizer, microphone):
    print("Recording audio...")
    with microphone as source:
        audio = recognizer.listen(source, phrase_time_limit=10)
    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_audio.close()
    with open(temp_audio.name, "wb") as f:
        f.write(audio.get_wav_data())
    print(f"Saved recording to {temp_audio.name}")
    return temp_audio.name

def transcribe_audio(audio_path, model_path=resource_path("models/ggml-base.en.bin")):
    print("Transcribing audio...")
    whisper_cli_path = resource_path("build/bin/Release/whisper-cli.exe")
    if not os.path.isfile(whisper_cli_path):
        print(f"whisper-cli.exe not found at {whisper_cli_path}")
        return ""
    cmd = [whisper_cli_path, "-m", model_path, "-f", audio_path, "--no-prints"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        transcription = re.sub(r"\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]\s*", "", result.stdout.strip())
        print(f"Transcription:\n{transcription}")
        return transcription
    except subprocess.CalledProcessError as e:
        print(f"Error during transcription: {e.stderr}")
        return ""

def perform_vision_task(question: str) -> str:
    global last_ocr_text
    print("Performing vision task with question:", question)
    with mss.mss() as sct:
        screenshot_path = sct.shot(output="screenshot.png")
    image = Image.open(screenshot_path)
    ocr_text = pytesseract.image_to_string(image, lang="rus+ukr+pol+eng").strip()

    print(f"OCR extracted:\n{ocr_text}\n")

    if not ocr_text:
        last_ocr_text = None
        return "I couldn't read any text on your screen."

    last_ocr_text = ocr_text

    # Optionally, you can return a summary or translation as before
    prompt = f"""You are an assistant that translates text extracted from a user's screen.
Here is the text extracted from the screen (which may be in Russian, Ukrainian, Polish, or Cyrillic script):

\"\"\"
{ocr_text}
\"\"\"

Please translate this text into English accurately.
"""
    return llm._call(prompt)

def speak_text(text, tts_engine):
    tts_engine.say(text)
    tts_engine.runAndWait()

# -------------------- MAIN LOOP --------------------
def main():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    tts_engine = pyttsx3.init()

    tts_engine.say("assistant starting")
    tts_engine.runAndWait()

    print("Press Ctrl+Shift+Alt+L at any time to quit.")
    try:
        while True:
            if keyboard.is_pressed('ctrl+shift+alt+l'):
                print("Keyboard shortcut detected. Quitting program.")
                break

            if listen_for_wake_word(recognizer, microphone):
                audio_path = record_audio(recognizer, microphone)
                transcription = transcribe_audio(audio_path)
                os.remove(audio_path)

                if transcription:
                    if "exit" in transcription.lower():
                        print("Voice command 'exit' detected. Quitting program.")
                        speak_text("Goodbye!", tts_engine)
                        break

                    if any(phrase in transcription.lower() for phrase in VISION_PHRASES):
                        response = perform_vision_task(transcription)
                    elif any(keyword in transcription.lower() for keyword in ["pdf", "document", "knowledge base"]):
                        print(f"Question for AI (retrieval): {transcription}")
                        result = qa_chain.invoke({"question": transcription})
                        response = result.get("answer", "").strip()
                    else:
                        print(f"Question for AI (chat): {transcription}")
                        # Add math keywords for context
                        math_keywords = ["solve", "calculate", "what is", "answer", "result", "equation"]
                        if last_ocr_text and (
                            any(word in transcription.lower() for word in ["screenshot", "screen", "image", "picture", "photo"])
                            or any(word in transcription.lower() for word in math_keywords)
                        ):
                            user_input = (
                                f"The following text was extracted from the last screenshot:\n"
                                f"{last_ocr_text}\n\n"
                                f"{transcription}"
                            )
                            response = ask_with_memory(user_input).strip()
                        else:
                            response = ask_with_memory(transcription).strip()
                    # TTS for all responses except exit
                    speak_text(response, tts_engine)
    except Exception as e:
        print("Unhandled exception:", e)
        input("Press Enter to exit...")  # Keeps the window open to read the error

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error at startup:", e)
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
