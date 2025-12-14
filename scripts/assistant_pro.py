import os
import sys
import time
import threading
import queue
import re
import tempfile
import json
import datetime

# --- Third Party Libraries ---
import cv2
import numpy as np
import mss
import pyautogui
import pygetwindow as gw
import pytesseract
import pyperclip
import pyttsx3
import speech_recognition as sr
import keyboard
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.spinner import Spinner

# --- LangChain / AI ---
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType

# -------------------- CONFIGURATION --------------------
# Point this to your Tesseract executable
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(__file__), relative_path)

pytesseract.pytesseract.tesseract_cmd = resource_path(r"Tesseract-OCR/tesseract.exe")

WAKE_WORDS = ["assistant", "jarvis", "computer", "help"]
console = Console()

# -------------------- 1. THE TOOLS (HANDS & EYES) --------------------
class AssistantTools:
    
    @staticmethod
    def get_active_window_info():
        """Returns title and geometry of the active window."""
        try:
            window = gw.getActiveWindow()
            if window:
                return {
                    "title": window.title,
                    "box": {"top": window.top, "left": window.left, "width": window.width, "height": window.height}
                }
        except:
            pass
        return None

    @staticmethod
    def preprocess_image(image):
        """Standardizes image for better OCR accuracy."""
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img

    @tool
    def read_screen(query: str = "") -> str:
        """
        Captures text from the currently active window. 
        Use this when the user asks about what is on their screen, code, or error messages.
        """
        console.log(f"[yellow]Tool Triggered:[/yellow] Reading Screen...")
        active_window = AssistantTools.get_active_window_info()
        
        with mss.mss() as sct:
            if active_window:
                monitor = active_window["box"]
                region_name = f"Active Window: {active_window['title']}"
            else:
                monitor = sct.monitors[1]
                region_name = "Full Screen"
            
            # Catch errors if window is off-screen
            try:
                sct_img = sct.grab(monitor)
            except:
                monitor = sct.monitors[1]
                sct_img = sct.grab(monitor)
                
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            
        # Preprocess and OCR
        processed_img = AssistantTools.preprocess_image(img)
        # PSM 6 is good for blocks of text/code
        text = pytesseract.image_to_string(processed_img, config='--psm 6').strip()
        
        if len(text) < 5:
            return "Screen appears empty or text is unreadable."
            
        return f"--- Content of {region_name} ---\n{text}\n----------------"

    @tool
    def get_clipboard(query: str = "") -> str:
        """
        Reads text currently copied to the system clipboard.
        Use this when user says 'summarize copied text' or 'what is in my clipboard'.
        """
        console.log(f"[yellow]Tool Triggered:[/yellow] Reading Clipboard...")
        content = pyperclip.paste()
        return f"Clipboard Content: {content}" if content else "Clipboard is empty."

    @tool
    def write_to_file(content: str) -> str:
        """
        Saves text to a file named 'notes.txt' on the desktop.
        Use this when the user asks to 'save this', 'take a note', or 'write this down'.
        """
        desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        filepath = os.path.join(desktop, 'ai_notes.txt')
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.datetime.now()}]\n{content}\n")
        return f"Saved to {filepath}"

# -------------------- 2. THE BRAIN (AGENT) --------------------
class AssistantBrain:
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.llm = ChatOllama(model="llama3", temperature=0.2)
        
        # Define the toolkit
        self.tools = [
            AssistantTools.read_screen,
            AssistantTools.get_clipboard,
            AssistantTools.write_to_file
        ]
        
        # Initialize the Agent (Structure that decides which tool to use)
        # 'structured-chat-zero-shot-react-description' is good for multi-tool use
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=False, # We will handle our own logging
            handle_parsing_errors=True
        )

    def think(self, user_input):
        try:
            # We wrap execution to handle the agent loop
            response = self.agent.run(user_input)
            return response
        except Exception as e:
            return f"I encountered an error while processing that: {str(e)}"

# -------------------- 3. THE INTERFACE (IO) --------------------
class AssistantIO:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.engine = pyttsx3.init()
        
        # Voice Settings
        voices = self.engine.getProperty('voices')
        # Try to find a good English voice (Index 1 is often female/better on Windows)
        try:
            self.engine.setProperty('voice', voices[1].id)
        except:
            pass
        self.engine.setProperty('rate', 160) # Slightly faster speech
    
    def speak(self, text):
        # Remove code blocks for speech
        clean_text = re.sub(r"```.*?```", "code block", text, flags=re.DOTALL)
        # Remove special chars
        clean_text = clean_text.replace("*", "")
        
        # Print to console nicely
        console.print(Panel(Text(text, style="cyan"), title="Assistant", border_style="blue"))
        
        self.engine.say(clean_text)
        self.engine.runAndWait()

    def listen_active(self):
        """Active listening mode (after wake word)."""
        console.print("[red]Listening for command...[/red]")
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                text = self.recognizer.recognize_google(audio).lower()
                console.print(f"[green]User:[/green] {text}")
                return text
            except sr.WaitTimeoutError:
                return None
            except sr.UnknownValueError:
                return None

    def listen_passive(self):
        """Passive listening for wake word (short buffer)."""
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                # Short timeout for quick loops
                audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=3)
                text = self.recognizer.recognize_google(audio).lower()
                if any(wake in text for wake in WAKE_WORDS):
                    return True
            except:
                pass
        return False

# -------------------- MAIN LOOP --------------------
def main():
    from PIL import Image # Lazy import
    
    # Init Systems
    console.clear()
    console.print(Panel.fit("[bold white]AI Assistant System Online[/bold white]", style="bold green"))
    
    brain = AssistantBrain()
    io = AssistantIO()
    
    io.speak("System initialized. I am listening.")
    
    while True:
        try:
            # Check for keyboard kill switch
            if keyboard.is_pressed('ctrl+shift+alt+q'):
                io.speak("Shutting down.")
                break

            # 1. Listen for Wake Word
            if io.listen_passive():
                io.speak("Yes?")
                
                # 2. Listen for Command
                command = io.listen_active()
                
                if command:
                    # check for exit command
                    if "exit" in command or "stop" in command:
                        io.speak("Going to sleep.")
                        continue
                    
                    # 3. Processing (The "Thinking" Phase)
                    with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
                        response = brain.think(command)
                    
                    # 4. Response
                    io.speak(response)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.log(f"[red]Error in main loop:[/red] {e}")

if __name__ == "__main__":
    main()