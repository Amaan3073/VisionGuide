import pyttsx3
import speech_recognition as sr
import threading
import queue
import time
from typing import Optional, Callable
from loguru import logger
import numpy as np
from dataclasses import dataclass
from enum import Enum
from functools import total_ordering

class AudioPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    EMERGENCY = 4

@total_ordering
@dataclass
class AudioMessage:
    text: str
    priority: AudioPriority
    timestamp: float
    callback: Optional[Callable] = None
    
    def __lt__(self, other):
        if not isinstance(other, AudioMessage):
            return NotImplemented
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.timestamp < other.timestamp
    
    def __eq__(self, other):
        if not isinstance(other, AudioMessage):
            return NotImplemented
        return (self.priority.value == other.priority.value and 
                self.timestamp == other.timestamp)

class AudioProcessor:
    def __init__(self):
        """Initialize audio processing system"""
        self.tts_engine = None
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Audio queues
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.audio_thread = None
        self.listening_thread = None
        
        # State
        self.running = False
        self.voice_commands_enabled = True
        
        # Thread-safe flag
        self.tts_ready = threading.Event()
        
        # Initialize components
        self._initialize_tts()
        self._initialize_speech_recognition()
        
        logger.info("Audio processor initialized")
    
    def _initialize_tts(self):
        """Initialize text-to-speech engine"""
        try:
            # Initialize TTS engine
            self.tts_engine = pyttsx3.init()
            
            # Configure voice properties
            voices = self.tts_engine.getProperty('voices')
            if voices:
                voice_index = min(1, len(voices) - 1)
                self.tts_engine.setProperty('voice', voices[voice_index].id)
            
            # Set speech rate and volume
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 1.0)
            
            # Set TTS ready flag
            self.tts_ready.set()
            
            logger.info("TTS engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            raise
    
    def _initialize_speech_recognition(self):
        """Initialize speech recognition"""
        try:
            with self.microphone as source:
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
            logger.info("Speech recognition initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize speech recognition: {e}")
    
    def speak_async(self, text: str, priority: AudioPriority = AudioPriority.NORMAL,
                   callback: Optional[Callable] = None):
        """Add text to speech queue for asynchronous playback"""
        if not text or text.strip() == "." or text.strip() == "":
            logger.debug("Skipping empty or minimal text")
            return
        
        message = AudioMessage(
            text=text,
            priority=priority,
            timestamp=time.time(),
            callback=callback
        )
        
        self.speech_queue.put(message)
        logger.info(f"Added to speech queue: {text}")
    
    def speak_immediately(self, text: str, interrupt: bool = False):
        """Speak text immediately using main thread"""
        if not text or text.strip() == "." or text.strip() == "":
            return
        
        try:
            # Wait for TTS to be ready
            if not self.tts_ready.wait(timeout=5):
                logger.error("TTS engine not ready")
                return
            
            if interrupt and self.is_speaking:
                self.tts_engine.stop()
            
            self.is_speaking = True
            logger.info(f"Speaking immediately: {text}")
            
            # Create a new TTS engine for immediate use
            temp_engine = pyttsx3.init()
            temp_engine.setProperty('rate', 150)
            temp_engine.setProperty('volume', 1.0)
            
            temp_engine.say(text)
            temp_engine.runAndWait()
            
            self.is_speaking = False
            logger.info("Immediate speech completed")
            
        except Exception as e:
            logger.error(f"Failed to speak text immediately: {e}")
            self.is_speaking = False
    
    def _speak_with_engine(self, text: str) -> bool:
        """Safely speak text using TTS engine"""
        try:
            # Create a new TTS engine for this thread
            engine = pyttsx3.init()
            
            # Configure the engine
            voices = engine.getProperty('voices')
            if voices:
                voice_index = min(1, len(voices) - 1)
                engine.setProperty('voice', voices[voice_index].id)
            
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            
            # Speak the text
            engine.say(text)
            engine.runAndWait()
            
            # Clean up
            engine.stop()
            del engine
            
            return True
            
        except Exception as e:
            logger.error(f"Speech engine error: {e}")
            return False
    
    def stop_speaking(self):
        """Stop current speech"""
        try:
            if self.tts_engine and self.is_speaking:
                self.tts_engine.stop()
                self.is_speaking = False
                logger.info("Speech stopped")
        except Exception as e:
            logger.error(f"Failed to stop speech: {e}")
    
    def _audio_worker(self):
        """Audio processing worker thread"""
        logger.info("Audio worker started")
        
        while self.running:
            try:
                # Get next message from queue
                try:
                    message = self.speech_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Skip if message is too old (except emergency)
                if (message.priority != AudioPriority.EMERGENCY and 
                    time.time() - message.timestamp > 15):
                    logger.debug("Skipping old message")
                    continue
                
                # Skip empty messages
                if not message.text or message.text.strip() == "." or message.text.strip() == "":
                    continue
                
                # Speak the message
                self.is_speaking = True
                logger.info(f"Speaking: {message.text}")
                
                # Use thread-safe speech method
                success = self._speak_with_engine(message.text)
                
                if success:
                    logger.info("Speech completed successfully")
                else:
                    logger.error("Speech failed")
                
                self.is_speaking = False
                
                # Call callback if provided
                if message.callback:
                    try:
                        message.callback()
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                # Mark task as done
                self.speech_queue.task_done()
                
                # Small delay to prevent audio overlap
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Audio worker error: {e}")
                self.is_speaking = False
                time.sleep(0.5)
        
        logger.info("Audio worker stopped")
    
    def start(self):
        """Start audio processing"""
        if self.running:
            return
        
        self.running = True
        
        # Start audio worker thread
        self.audio_thread = threading.Thread(target=self._audio_worker)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        # Start voice command listening thread
        if self.voice_commands_enabled:
            self.listening_thread = threading.Thread(target=self._listen_for_commands)
            self.listening_thread.daemon = True
            self.listening_thread.start()
        
        logger.info("Audio processor started")
    
    def stop(self):
        """Stop audio processing"""
        self.running = False
        
        # Stop current speech
        self.stop_speaking()
        
        # Wait for threads to finish
        if self.audio_thread:
            self.audio_thread.join(timeout=5)
        
        if self.listening_thread:
            self.listening_thread.join(timeout=5)
        
        logger.info("Audio processor stopped")
    
    def _listen_for_commands(self):
        """Listen for voice commands"""
        logger.info("Voice command listener started")
        
        while self.running:
            try:
                # Don't listen while speaking
                if self.is_speaking:
                    time.sleep(0.1)
                    continue
                
                with self.microphone as source:
                    logger.debug("Listening for voice command...")
                    audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=3)
                
                # Recognize speech
                try:
                    command = self.recognizer.recognize_google(audio, language='en-US').lower()
                    logger.info(f"Voice command detected: '{command}'")
                    self._process_voice_command(command)
                    
                except sr.UnknownValueError:
                    logger.debug("Could not understand audio")
                    pass
                except sr.RequestError as e:
                    logger.error(f"Speech recognition service error: {e}")
                    time.sleep(1)
                    
            except sr.WaitTimeoutError:
                logger.debug("No speech detected")
                pass
            except Exception as e:
                logger.error(f"Voice command listening error: {e}")
                time.sleep(1)
        
        logger.info("Voice command listener stopped")
    
    def _process_voice_command(self, command: str):
        """Process recognized voice command"""
        logger.info(f"Processing voice command: {command}")
        
        if "what do you see" in command or "describe scene" in command:
            self.speak_immediately("Analyzing current scene", interrupt=True)
            
        elif "stop talking" in command or "be quiet" in command:
            self.stop_speaking()
            self._clear_speech_queue()
            self.speak_immediately("Okay, I'll be quiet")
            
        elif "repeat" in command:
            self.speak_immediately("Repeating last description")
            
        elif "help" in command or "emergency" in command:
            self.speak_immediately("Emergency mode activated", interrupt=True)
            
        elif "where am i" in command or "location" in command:
            self.speak_immediately("Getting your current location")
    
    def _clear_speech_queue(self):
        """Clear speech queue"""
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
    
    def _adjust_volume(self, change: float):
        """Adjust TTS volume"""
        try:
            current_volume = self.tts_engine.getProperty('volume')
            new_volume = max(0.0, min(1.0, current_volume + change))
            self.tts_engine.setProperty('volume', new_volume)
            self.speak_immediately(f"Volume set to {int(new_volume * 100)} percent")
        except Exception as e:
            logger.error(f"Failed to adjust volume: {e}")
    
    def _adjust_speech_rate(self, change: int):
        """Adjust TTS speech rate"""
        try:
            current_rate = self.tts_engine.getProperty('rate')
            new_rate = max(50, min(300, current_rate + change))
            self.tts_engine.setProperty('rate', new_rate)
            self.speak_immediately(f"Speech rate adjusted")
        except Exception as e:
            logger.error(f"Failed to adjust speech rate: {e}")
    
    def test_audio(self):
        """Test audio output"""
        logger.info("Testing audio output...")
        self.speak_immediately("VisionGuide AI audio test. If you can hear this, audio is working correctly.")

    
    def _clear_non_emergency_queue(self):
        """Clear speech queue of non-emergency messages"""
        new_queue = queue.PriorityQueue()
        
        while not self.speech_queue.empty():
            try:
                priority, message = self.speech_queue.get_nowait()
                if message.priority == AudioPriority.EMERGENCY:
                    new_queue.put((priority, message))
            except queue.Empty:
                break
        
        self.speech_queue = new_queue
    
    def set_voice_properties(self, rate: int = None, volume: float = None):
        """
        Set voice properties
        
        Args:
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
        """
        if rate is not None:
            self.tts_engine.setProperty('rate', rate)
        
        if volume is not None:
            self.tts_engine.setProperty('volume', volume)
    
    def get_available_voices(self) -> list:
        """Get list of available voices"""
        voices = self.tts_engine.getProperty('voices')
        return [(i, voice.name) for i, voice in enumerate(voices)]
    
    def set_voice(self, voice_index: int):
        """
        Set voice by index
        
        Args:
            voice_index: Index of voice to use
        """
        voices = self.tts_engine.getProperty('voices')
        if 0 <= voice_index < len(voices):
            self.tts_engine.setProperty('voice', voices[voice_index].id)
            logger.info(f"Voice changed to: {voices[voice_index].name}")
        else:
            logger.warning(f"Invalid voice index: {voice_index}")

# Audio utilities
def create_spatial_audio_description(objects: list, user_facing: str = "north") -> str:
    """
    Create spatial audio description
    
    Args:
        objects: List of detected objects with position info
        user_facing: Direction user is facing
        
    Returns:
        Spatial audio description
    """
    descriptions = []
    
    for obj in objects:
        # Convert object position to spatial description
        direction = obj.get('direction', 'unknown')
        distance = obj.get('distance', 0)
        name = obj.get('name', 'object')
        
        # Create natural spatial description
        if direction == 'center':
            spatial_desc = f"{name} directly ahead"
        elif direction == 'left':
            spatial_desc = f"{name} to your left"
        elif direction == 'right':
            spatial_desc = f"{name} to your right"
        else:
            spatial_desc = f"{name} nearby"
        
        if distance > 0:
            if distance < 2:
                spatial_desc += " very close"
            elif distance < 5:
                spatial_desc += f" {distance:.0f} steps away"
            else:
                spatial_desc += " far away"
        
        descriptions.append(spatial_desc)
    
    return ". ".join(descriptions)

def format_navigation_instruction(instruction: dict) -> str:
    """
    Format navigation instruction for audio
    
    Args:
        instruction: Navigation instruction dict
        
    Returns:
        Formatted audio instruction
    """
    direction = instruction.get('direction', '')
    distance = instruction.get('distance', 0)
    landmark = instruction.get('landmark', '')
    
    if direction == 'straight':
        base = "Continue straight"
    elif direction == 'left':
        base = "Turn left"
    elif direction == 'right':
        base = "Turn right"
    else:
        base = "Continue"
    
    if distance > 0:
        if distance < 10:
            base += f" for {distance:.0f} steps"
        else:
            base += f" for {distance:.0f} meters"
    
    if landmark:
        base += f" towards {landmark}"
    
    return base
