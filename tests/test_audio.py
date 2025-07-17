import pyttsx3
import time

def test_tts():
    print("Testing TTS engine...")
    
    try:
        engine = pyttsx3.init()
        
        # Get available voices
        voices = engine.getProperty('voices')
        print(f"Available voices: {len(voices)}")
        
        # Configure voice
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        
        # Test speech
        engine.say("VisionGuide AI audio test. If you can hear this, TTS is working correctly.")
        engine.runAndWait()
        
        print("TTS test completed successfully")
        return True
        
    except Exception as e:
        print(f"TTS test failed: {e}")
        return False

if __name__ == "__main__":
    test_tts()
