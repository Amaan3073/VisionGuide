import cv2
import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from main import VisionGuideAI

def test_basic_functionality():
    """Test basic VisionGuide AI functionality"""
    print("Testing VisionGuide AI...")
    
    # Initialize system
    vision_guide = VisionGuideAI()
    
    try:
        # Test camera
        if not vision_guide.start_camera():
            print("❌ Camera test failed")
            return False
        
        print("✅ Camera initialized")
        
        # Test single frame processing
        ret, frame = vision_guide.camera.read()
        if ret:
            description = vision_guide.process_frame(frame)
            print(f"✅ Frame processed: {description}")
        else:
            print("❌ Frame capture failed")
            return False
        
        # Test audio
        vision_guide.audio_processor.start()
        vision_guide.audio_processor.speak_immediately("VisionGuide AI is working correctly")
        
        print("✅ Audio test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        vision_guide.stop()

if __name__ == "__main__":
    test_basic_functionality()
