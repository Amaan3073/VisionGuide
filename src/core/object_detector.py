import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
import time

@dataclass
class DetectedObject:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center_point: Tuple[int, int]
    area: float
    distance: Optional[float] = None
    direction: Optional[str] = None

class ObjectDetector:
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "auto"):
        """
        Initialize YOLO object detector
        
        Args:
            model_path: Path to YOLO model
            device: Computing device ('cpu', 'cuda', 'auto')
        """
        self.device = self._get_device(device)
        self.model = self._load_model(model_path)
        self.class_names = self.model.names
        logger.info(f"Object detector initialized on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best available device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self, model_path: str) -> YOLO:
        """Load YOLO model"""
        try:
            model = YOLO(model_path)
            model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect_objects(self, frame: np.ndarray, 
                      confidence_threshold: float = 0.5,
                      iou_threshold: float = 0.45) -> List[DetectedObject]:
        """
        Detect objects in frame
        
        Args:
            frame: Input image frame
            confidence_threshold: Minimum confidence for detection
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of detected objects
        """
        try:
            # Run inference
            results = self.model(frame, 
                               conf=confidence_threshold,
                               iou=iou_threshold,
                               verbose=False)
            
            detected_objects = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box information
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Calculate center point and area
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        area = (x2 - x1) * (y2 - y1)
                        
                        # Get class name
                        class_name = self.class_names[class_id]
                        
                        # Calculate direction relative to frame center
                        frame_center_x = frame.shape[1] // 2
                        direction = self._calculate_direction(center_x, frame_center_x)
                        
                        detected_obj = DetectedObject(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=float(confidence),
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            center_point=(center_x, center_y),
                            area=area,
                            direction=direction
                        )
                        
                        detected_objects.append(detected_obj)
            
            return detected_objects
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    def _calculate_direction(self, object_center_x: int, frame_center_x: int) -> str:
        """Calculate object direction relative to frame center"""
        diff = object_center_x - frame_center_x
        threshold = 50  # pixels
        
        if abs(diff) < threshold:
            return "center"
        elif diff > 0:
            return "right"
        else:
            return "left"
    
    def draw_detections(self, frame: np.ndarray, 
                       detected_objects: List[DetectedObject]) -> np.ndarray:
        """
        Draw detection boxes and labels on frame
        
        Args:
            frame: Input frame
            detected_objects: List of detected objects
            
        Returns:
            Frame with drawn detections
        """
        for obj in detected_objects:
            x1, y1, x2, y2 = obj.bbox
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label
            label = f"{obj.class_name}: {obj.confidence:.2f}"
            if obj.distance:
                label += f" ({obj.distance:.1f}m)"
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(frame, (x1, y1 - label_height - 10), 
                         (x1 + label_width, y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame

class PersonalizedObjectDetector:
    """Extended detector for personalized objects (faces, pets, etc.)"""
    
    def __init__(self):
        self.known_faces = {}  # name -> face_encoding
        self.known_objects = {}  # name -> object_features
        self.face_recognition_available = False
        
        # Try to import face_recognition
        try:
            import face_recognition
            self.face_recognition_available = True
            logger.info("Face recognition module available")
        except ImportError:
            logger.warning("Face recognition module not available - skipping face recognition features")
        
    def add_known_face(self, name: str, face_image: np.ndarray):
        """Add a known face to the database"""
        if not self.face_recognition_available:
            logger.warning("Face recognition not available")
            return False
            
        try:
            import face_recognition
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(face_image)
            if face_locations:
                face_encodings = face_recognition.face_encodings(face_image, face_locations)
                if face_encodings:
                    self.known_faces[name] = face_encodings[0]
                    logger.info(f"Added face for {name}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to add face for {name}: {e}")
            return False
    
    def recognize_faces(self, frame: np.ndarray) -> List[Dict]:
        """Recognize known faces in frame"""
        if not self.face_recognition_available:
            return []
            
        try:
            import face_recognition
            
            # Find all face locations and encodings
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            
            recognized_faces = []
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare with known faces
                name = "Unknown"
                for known_name, known_encoding in self.known_faces.items():
                    matches = face_recognition.compare_faces([known_encoding], face_encoding)
                    if matches[0]:
                        name = known_name
                        break
                
                recognized_faces.append({
                    'name': name,
                    'location': (left, top, right, bottom),
                    'center': ((left + right) // 2, (top + bottom) // 2)
                })
            
            return recognized_faces
            
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return []
