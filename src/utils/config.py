from pydantic import BaseModel
from typing import List, Dict, Optional
import os
from pathlib import Path

class ModelConfig(BaseModel):
    yolo_model_path: str = "yolov8n.pt"
    depth_model_type: str = "MiDaS_small"
    face_recognition_model: str = "models/face_recognition"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45

class AudioConfig(BaseModel):
    tts_engine: str = "pyttsx3"
    speech_rate: int = 200
    volume: float = 0.9
    voice_index: int = 0

class NavigationConfig(BaseModel):
    api_key: Optional[str] = None
    max_walking_distance: float = 5000.0  # meters
    step_size: float = 0.75  # meters per step

class EmergencyConfig(BaseModel):
    emergency_contacts: List[str] = []
    emergency_contacts: List[str] = []
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_phone_number: Optional[str] = None

class AppConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    audio: AudioConfig = AudioConfig()
    navigation: NavigationConfig = NavigationConfig()
    emergency: EmergencyConfig = EmergencyConfig()
    
    # Directories
    project_root: Path = Path(__file__).parent.parent.parent
    models_dir: Path = project_root / "models"
    data_dir: Path = project_root / "data"
    
    # Camera settings
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30

    # Debug settings - ADD THIS
    debug_mode: bool = True  # Set to True for debugging with visual output

# Global configuration instance
config = AppConfig()
