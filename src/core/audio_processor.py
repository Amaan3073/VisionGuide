import pyttsx3
import speech_recognition as sr
import threading
import queue
import time
import subprocess
from typing import Optional, Callable
from loguru import logger
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
        
        # State
        self.running = False
        self.voice_commands_enabled = True
        
        # Initialize components
        self._initialize_tts()
        self._initialize_speech_recognition()
        
        logger.info("Audio processor initialized")
    
    def _initialize_tts(self):
        """Initialize text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure voice properties
            voices = self.tts_engine.getProperty('voices')
            if voices:
                voice_index = min(1, len(voices) - 1)
                self.tts_engine.setProperty('voice', voices[voice_index].id)
            
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 1.0)
            
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
            return
        
        message = AudioMessage(
            text=text,
            priority=priority,
            timestamp=time.time(),
            callback=callback
        )
        
        self.speech_queue.put(message)
    
    def speak_immediately(self, text: str, interrupt: bool = False):
        """Speak text immediately using Windows SAPI"""
        if not text or text.strip() == "." or text.strip() == "":
            return
        
        try:
            self.is_speaking = True
            logger.info(f"Speaking: {text}")
            
            # Use Windows SAPI for immediate speech
            self._speak_with_sapi(text)
            
            self.is_speaking = False
            
        except Exception as e:
            logger.error(f"Failed to speak text: {e}")
            self.is_speaking = False
    
    def _speak_with_sapi(self, text: str) -> bool:
        """Use Windows SAPI for TTS"""
        try:
            # Escape text for PowerShell
            escaped_text = text.replace('"', '""').replace("'", "''")
            
            # Create PowerShell command
            ps_command = f'''
            Add-Type -AssemblyName System.Speech
            $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
            $synth.Rate = 0
            $synth.Volume = 100
            $synth.Speak("{escaped_text}")
            '''
            
            # Execute PowerShell command
            result = subprocess.run([
                'powershell', '-Command', ps_command
            ], capture_output=True, text=True, timeout=15)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"SAPI TTS error: {e}")
            return False
    
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
                
                # Skip if message is too old
                if time.time() - message.timestamp > 15:
                    continue
                
                # Skip empty messages
                if not message.text or message.text.strip() == "." or message.text.strip() == "":
                    continue
                
                # Speak the message
                self.is_speaking = True
                logger.info(f"Speaking: {message.text}")
                
                # Use Windows SAPI for speech
                self._speak_with_sapi(message.text)
                
                self.is_speaking = False
                
                # Call callback if provided
                if message.callback:
                    try:
                        message.callback()
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                # Mark task as done
                self.speech_queue.task_done()
                
                # Small delay
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Audio worker error: {e}")
                self.is_speaking = False
                time.sleep(0.5)
        
        logger.info("Audio worker stopped")
    
    def listen_for_command(self) -> str:
        """Listen for a single voice command (push-to-talk)"""
        try:
            logger.info("Listening for voice command...")
            
            with self.microphone as source:
                # Listen for audio
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
            # Recognize speech
            command = self.recognizer.recognize_google(audio, language='en-US').lower()
            logger.info(f"Voice command detected: '{command}'")
            
            return command
            
        except sr.WaitTimeoutError:
            logger.info("No speech detected within timeout")
            return ""
        except sr.UnknownValueError:
            logger.info("Could not understand audio")
            return ""
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            return ""
        except Exception as e:
            logger.error(f"Voice command error: {e}")
            return ""
    
    def process_voice_command(self, command: str) -> str:
        """Process voice command and return response"""
        if not command:
            return ""
        
        logger.info(f"Processing command: {command}")
        
        if "what do you see" in command or "describe scene" in command:
            return "describe_scene"
            
        elif "stop talking" in command or "be quiet" in command:
            self.stop_speaking()
            self.speak_immediately("Okay, I'll be quiet")
            return "stop_talking"
            
        elif "repeat" in command or "say again" in command:
            return "repeat_last"
            
        elif "help" in command or "emergency" in command:
            return "emergency"
            
        elif "where am i" in command or "location" in command:
            return "get_location"
            
        elif "volume up" in command:
            self.speak_immediately("Volume increased")
            return "volume_up"
            
        elif "volume down" in command:
            self.speak_immediately("Volume decreased")
            return "volume_down"
            
        else:
            self.speak_immediately("I didn't understand that command")
            return "unknown"
    
    def stop_speaking(self):
        """Stop current speech"""
        try:
            # Clear the speech queue
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.is_speaking = False
            logger.info("Speech stopped and queue cleared")
            
        except Exception as e:
            logger.error(f"Failed to stop speech: {e}")
    
    def start(self):
        """Start audio processing"""
        if self.running:
            return
        
        self.running = True
        
        # Start audio worker thread
        self.audio_thread = threading.Thread(target=self._audio_worker)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        logger.info("Audio processor started")
    
    def stop(self):
        """Stop audio processing"""
        self.running = False
        
        # Stop current speech
        self.stop_speaking()
        
        # Wait for thread to finish
        if self.audio_thread:
            self.audio_thread.join(timeout=5)
        
        logger.info("Audio processor stopped")

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
        self.speak_immediately("VisionGuide AI audio test. Audio is working correctly.")


    
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
