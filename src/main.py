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
from core.scene_tracker import SceneTracker
from core.audio_processor import AudioProcessor, AudioPriority
from utils.config import config

class VisionGuideAI:
    def __init__(self):
        """Initialize VisionGuide AI system"""
        self.running = False
        self.camera = None
        self.should_exit = False
        self.last_description = ""
        
        # Initialize components
        self.object_detector = ObjectDetector(config.model.yolo_model_path)
        self.depth_estimator = DepthEstimator(model_type=config.model.depth_model_type)
        self.scene_analyzer = SceneAnalyzer()
        self.scene_tracker = SceneTracker()  # NEW: Smart scene tracker
        self.audio_processor = AudioProcessor()
        self.personalized_detector = PersonalizedObjectDetector()
        
        # Threading
        self.detection_thread = None
        
        # Smart announcement settings
        self.auto_announce_changes = True
        self.periodic_summary_interval = 30.0  # Summary every 30 seconds
        self.last_summary_time = 0
        
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
        """Main detection loop with smart announcements"""
        
        # Create window
        cv2.namedWindow('VisionGuide AI', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('VisionGuide AI', config.frame_width, config.frame_height)
        
        while self.running and not self.should_exit:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
                # Detect objects
                detected_objects = self.object_detector.detect_objects(
                    frame, 
                    config.model.confidence_threshold,
                    config.model.iou_threshold
                )
                
                # Estimate depth for detected objects
                depth_map = self.depth_estimator.estimate_depth(frame)
                for obj in detected_objects:
                    obj.distance = self.depth_estimator.get_object_distance(
                        depth_map, obj.bbox, config.navigation.step_size
                    )
                
                # Update scene tracker and get changes
                changes = self.scene_tracker.update_scene(detected_objects)
                
                # Announce meaningful changes only
                if changes and self.auto_announce_changes:
                    if self.scene_tracker.should_announce():
                        # Filter out rapid changes and only keep meaningful ones
                        meaningful_changes = []
                        for change_type, message in changes.items():
                            if message and not self._is_rapid_change(change_type):
                                meaningful_changes.append(message)
                        
                        if meaningful_changes:
                            # Limit to 2 most important changes
                            announcement = ". ".join(meaningful_changes[:2])
                            self.audio_processor.speak_async(announcement, AudioPriority.HIGH)
                            self.last_description = announcement
                            logger.info(f"Smart announcement: {announcement}")
                
                # Periodic scene summary (less frequent)
                current_time = time.time()
                if current_time - self.last_summary_time >= self.periodic_summary_interval:
                    summary = self.scene_tracker.get_scene_summary()
                    if summary and "analyzing" not in summary.lower():
                        self.audio_processor.speak_async(f"Scene update: {summary}", AudioPriority.LOW)
                        logger.info(f"Scene summary: {summary}")
                    self.last_summary_time = current_time
                
                # Display frame with detections
                display_frame = self.object_detector.draw_detections(
                    frame.copy(), detected_objects
                )
                
                # Add distance information to display
                for obj in detected_objects:
                    if obj.distance:
                        x1, y1, x2, y2 = obj.bbox
                        distance_text = f"{obj.distance:.1f} steps"
                        cv2.putText(display_frame, distance_text, 
                                (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Add control instructions
                instructions = [
                    "VisionGuide AI - Smart Mode:",
                    "Q - Quit",
                    "S - Manual description",
                    "V - Voice command",
                    "R - Repeat last",
                    "T - Test audio",
                    "A - Toggle auto-announce",
                    "C - Current scene summary",
                    "SPACE - Stop talking"
                ]
                
                y_offset = 30
                for instruction in instructions:
                    cv2.putText(display_frame, instruction, 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_offset += 25
                
                # Add mode indicator
                mode_text = "AUTO-ANNOUNCE: ON" if self.auto_announce_changes else "AUTO-ANNOUNCE: OFF"
                cv2.putText(display_frame, mode_text, 
                           (10, display_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Add speaking indicator
                if self.audio_processor.is_speaking:
                    cv2.putText(display_frame, "SPEAKING...", 
                               (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow('VisionGuide AI', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    logger.info("Quit key pressed")
                    self.should_exit = True
                    break
                    
                elif key == ord('s') or key == ord('S'):
                    # Manual scene description
                    audio_description = self.process_frame(frame)
                    if audio_description and audio_description.strip() != ".":
                        self.last_description = audio_description
                        self.audio_processor.speak_async(audio_description, AudioPriority.HIGH)
                        logger.info("Manual description triggered")
                        
                elif key == ord('v') or key == ord('V'):
                    # Voice command mode
                    self.handle_voice_command(frame)
                    
                elif key == ord('r') or key == ord('R'):
                    # Repeat last description
                    if self.last_description:
                        self.audio_processor.speak_async(self.last_description, AudioPriority.HIGH)
                        logger.info("Repeating last description")
                    else:
                        self.audio_processor.speak_immediately("No previous description to repeat")
                        
                elif key == ord('t') or key == ord('T'):
                    # Test audio
                    self.audio_processor.test_audio()
                    
                elif key == ord('a') or key == ord('A'):
                    # Toggle auto-announce
                    self.auto_announce_changes = not self.auto_announce_changes
                    status = "enabled" if self.auto_announce_changes else "disabled"
                    self.audio_processor.speak_immediately(f"Auto-announce {status}")
                    logger.info(f"Auto-announce {status}")
                    
                elif key == ord('c') or key == ord('C'):
                    # Current scene summary
                    summary = self.scene_tracker.get_scene_summary()
                    self.audio_processor.speak_async(summary, AudioPriority.HIGH)
                    logger.info(f"Manual scene summary: {summary}")
                    
                elif key == ord(' '):  # Space key
                    # Stop talking
                    self.audio_processor.stop_speaking()
                    logger.info("Speech stopped by user")
                
                # Small delay
                time.sleep(0.03)
                    
            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                time.sleep(0.1)
        
        # Clean up
        cv2.destroyAllWindows()

    def _is_rapid_change(self, change_type: str) -> bool:
        """Check if this is a rapid change that should be filtered"""
        # Filter out rapid appearance/disappearance cycles
        return "gone_" in change_type or "new_" in change_type

    
    def handle_voice_command(self, frame: np.ndarray):
        """Handle voice command input"""
        try:
            # Show listening indicator
            self.audio_processor.speak_immediately("Listening for command")
            
            # Listen for command
            command = self.audio_processor.listen_for_command()
            
            if command:
                # Process command
                action = self.audio_processor.process_voice_command(command)
                
                # Handle specific actions
                if action == "describe_scene":
                    summary = self.scene_tracker.get_scene_summary()
                    self.audio_processor.speak_async(summary, AudioPriority.HIGH)
                    self.last_description = summary
                        
                elif action == "repeat_last":
                    if self.last_description:
                        self.audio_processor.speak_async(self.last_description, AudioPriority.HIGH)
                    else:
                        self.audio_processor.speak_immediately("No previous description to repeat")
                        
                elif action == "emergency":
                    self.audio_processor.speak_immediately("Emergency mode activated. This feature will be implemented soon.")
                    
                elif action == "get_location":
                    self.audio_processor.speak_immediately("Location services will be implemented soon.")
                    
            else:
                self.audio_processor.speak_immediately("No command heard")
                
        except Exception as e:
            logger.error(f"Voice command error: {e}")
            self.audio_processor.speak_immediately("Voice command failed")
    
    def start(self):
        """Start the VisionGuide AI system"""
        if not self.start_camera():
            return False
        
        self.running = True
        self.should_exit = False
        
        # Start audio processor
        self.audio_processor.start()
        
        # Test audio
        self.audio_processor.test_audio()
        time.sleep(2)
        
        # Welcome message
        self.audio_processor.speak_immediately("VisionGuide AI Smart Mode is ready. I will only announce when things change.")
        
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
        
        # Wait for thread to finish
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
            logger.info("VisionGuide AI Smart Mode is running.")
            
            # Keep main thread alive
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
