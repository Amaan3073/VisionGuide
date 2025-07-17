import cv2
import numpy as np
from typing import Optional
import threading
import time
from loguru import logger
import sys
from pathlib import Path
import signal

# Add src to path
sys.path.append(str(Path(__file__).parent))

from core.object_detector import ObjectDetector, PersonalizedObjectDetector
from core.depth_estimator import DepthEstimator
from core.scene_analyzer import SceneAnalyzer, format_for_audio
from core.audio_processor import AudioProcessor, AudioPriority
from utils.config import config

class VisionGuideAI:
    def __init__(self):
        """Initialize VisionGuide AI system"""
        self.running = False
        self.camera = None
        self.should_exit = False
        
        # Initialize components
        self.object_detector = ObjectDetector(config.model.yolo_model_path)
        self.depth_estimator = DepthEstimator(model_type=config.model.depth_model_type)
        self.scene_analyzer = SceneAnalyzer()
        self.audio_processor = AudioProcessor()
        self.personalized_detector = PersonalizedObjectDetector()
        
        # Threading
        self.detection_thread = None
        self.audio_thread = None
        
        logger.info("VisionGuide AI initialized successfully")
    
    def start_camera(self) -> bool:
        """Start camera capture"""
        try:
            self.camera = cv2.VideoCapture(config.camera_index)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, config.fps)
            
            if not self.camera.isOpened():
                logger.error("Failed to open camera")
                return False
            
            logger.info("Camera started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        if self.camera:
            self.camera.release()
            self.camera = None
            logger.info("Camera stopped")
    
    def process_frame(self, frame: np.ndarray) -> str:
        """Process single frame and return audio description"""
        try:
            # Detect objects
            detected_objects = self.object_detector.detect_objects(
                frame, 
                config.model.confidence_threshold,
                config.model.iou_threshold
            )
            
            # Estimate depth
            depth_map = self.depth_estimator.estimate_depth(frame)
            
            # Calculate distances for detected objects
            for obj in detected_objects:
                obj.distance = self.depth_estimator.get_object_distance(
                    depth_map, obj.bbox, config.navigation.step_size
                )
            
            # Recognize known faces
            recognized_faces = self.personalized_detector.recognize_faces(frame)
            
            # Analyze scene
            scene_context = self.scene_analyzer.analyze_scene(detected_objects, depth_map)
            
            # Format for audio
            audio_description = format_for_audio(scene_context, detected_objects)
            
            # Add personalized recognitions
            if recognized_faces:
                face_descriptions = []
                for face in recognized_faces:
                    if face['name'] != "Unknown":
                        face_descriptions.append(f"{face['name']} is present")
                if face_descriptions:
                    audio_description = ". ".join(face_descriptions) + ". " + audio_description
            
            return audio_description
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return "Unable to process scene"
    
    def run_detection_loop(self):
        """Main detection loop"""
        last_announcement = time.time()
        announcement_interval = 4.0  # seconds
        
        # Create window if debug mode is enabled
        if config.debug_mode:
            cv2.namedWindow('VisionGuide AI', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('VisionGuide AI', config.frame_width, config.frame_height)
        
        while self.running and not self.should_exit:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
                # Process frame periodically for audio
                current_time = time.time()
                if current_time - last_announcement >= announcement_interval:
                    audio_description = self.process_frame(frame)
                    
                    # Skip empty descriptions
                    if audio_description and audio_description.strip() != "." and audio_description.strip() != "":
                        logger.info(f"Processing audio: {audio_description}")
                        
                        # Queue audio for playback
                        self.audio_processor.speak_async(audio_description, AudioPriority.NORMAL)
                        
                        # Also log to console for debugging
                        print(f"🔊 AUDIO: {audio_description}")
                    
                    last_announcement = current_time
                
                # Display frame with detections (if debug mode)
                if config.debug_mode:
                    detected_objects = self.object_detector.detect_objects(
                        frame, 
                        config.model.confidence_threshold,
                        config.model.iou_threshold
                    )
                    
                    # Draw detections
                    display_frame = self.object_detector.draw_detections(
                        frame.copy(), detected_objects
                    )
                    
                    # Add text overlay
                    cv2.putText(display_frame, "VisionGuide AI - Press 'q' to quit", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add speaking indicator
                    if self.audio_processor.is_speaking:
                        cv2.putText(display_frame, "SPEAKING...", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    cv2.imshow('VisionGuide AI', display_frame)
                
                # Check for keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    logger.info("Quit key pressed")
                    self.should_exit = True
                    break
                elif key == ord('s') or key == ord('S'):
                    # Manual scene description
                    audio_description = self.process_frame(frame)
                    if audio_description and audio_description.strip() != ".":
                        logger.info("Manual scene description triggered")
                        self.audio_processor.speak_async(audio_description, AudioPriority.HIGH)
                elif key == ord('t') or key == ord('T'):
                    # Test audio
                    self.audio_processor.test_audio()
                
                # Small delay
                time.sleep(0.03)
                    
            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                time.sleep(0.1)
        
        # Clean up
        if config.debug_mode:
            cv2.destroyAllWindows()
    
    def start(self):
        """Start the VisionGuide AI system"""
        if not self.start_camera():
            return False
        
        self.running = True
        self.should_exit = False
        
        # Start audio processor
        self.audio_processor.start()
        
        # Test audio first
        time.sleep(1)
        logger.info("Testing audio system...")
        self.audio_processor.test_audio()
        
        # Welcome message
        time.sleep(2)
        self.audio_processor.speak_async("VisionGuide AI is ready. I will describe what I see every few seconds.", AudioPriority.HIGH)
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.run_detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        logger.info("VisionGuide AI started")
        return True
    
    def stop(self):
        """Stop the VisionGuide AI system"""
        self.running = False
        self.should_exit = True
        
        # Goodbye message
        self.audio_processor.speak_immediately("VisionGuide AI is shutting down. Goodbye.")
        
        # Wait for threads to finish
        if self.detection_thread:
            self.detection_thread.join(timeout=5)
        
        # Stop camera
        self.stop_camera()
        
        # Stop audio
        self.audio_processor.stop()
        
        # Close windows
        cv2.destroyAllWindows()
        
        logger.info("VisionGuide AI stopped")
    
    def add_known_person(self, name: str, image_path: str) -> bool:
        """Add a known person to the system"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return False
            
            return self.personalized_detector.add_known_face(name, image)
            
        except Exception as e:
            logger.error(f"Failed to add person {name}: {e}")
            return False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("Received interrupt signal")
    global vision_guide
    if vision_guide:
        vision_guide.stop()
    sys.exit(0)

def main():
    """Main function"""
    global vision_guide
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize logging
    logger.add("logs/visionguide.log", rotation="10 MB")
    
    # Create VisionGuide AI instance
    vision_guide = VisionGuideAI()
    
    try:
        # Start the system
        if vision_guide.start():
            logger.info("VisionGuide AI is running. Press 'q' to quit, 's' for manual description, 't' to test audio.")
            
            # Simple input loop
            while vision_guide.running and not vision_guide.should_exit:
                try:
                    time.sleep(0.1)
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt received")
                    break
        else:
            logger.error("Failed to start VisionGuide AI")
    
    except Exception as e:
        logger.error(f"Main loop error: {e}")
    
    finally:
        # Clean shutdown
        vision_guide.stop()
        logger.info("VisionGuide AI shutdown complete")

if __name__ == "__main__":
    main()
