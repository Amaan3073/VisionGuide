import cv2
import torch
import numpy as np
from typing import Tuple, Optional
import urllib.request
import os
from loguru import logger

class DepthEstimator:
    def __init__(self, model_type: str = "MiDaS_small"):
        """Initialize depth estimation model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model = None
        self.transform = None
        
        # Calibration parameters for distance estimation
        self.baseline_distance = 2.0  # meters (known distance for calibration)
        self.baseline_depth_value = 0.5  # corresponding depth value
        self.min_distance = 0.5  # minimum distance in meters
        self.max_distance = 10.0  # maximum distance in meters
        
        self._load_model()
        
    def _load_model(self):
        """Load MiDaS depth estimation model"""
        try:
            # Load MiDaS model
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
            self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            
            if "small" in self.model_type.lower():
                self.transform = midas_transforms.small_transform
            elif "DPT" in self.model_type:
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.default_transform
                
            logger.info(f"Depth estimation model loaded: {self.model_type}")
            
        except Exception as e:
            logger.error(f"Failed to load depth model: {e}")
            self.model = None
            self.transform = None
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth map from input frame"""
        if self.model is None or self.transform is None:
            return np.ones(frame.shape[:2]) * 2.0  # Return default depth
        
        try:
            # Convert BGR to RGB
            if len(frame.shape) == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame
            
            # Apply transforms
            input_tensor = self.transform(rgb_frame).to(self.device)
            
            # Predict depth
            with torch.no_grad():
                prediction = self.model(input_tensor)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=rgb_frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # Convert to numpy
            depth_map = prediction.cpu().numpy()
            
            return depth_map
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return np.ones(frame.shape[:2]) * 2.0
    
    def get_object_distance(self, depth_map: np.ndarray, 
                           bbox: Tuple[int, int, int, int],
                           step_size: float = 0.75) -> float:
        """Calculate object distance from depth map"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Ensure bbox is within image bounds
            h, w = depth_map.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return 2.0  # Default distance
            
            # Extract depth values in bounding box
            roi_depth = depth_map[y1:y2, x1:x2]
            
            # Use median of central region (more stable)
            center_h, center_w = roi_depth.shape
            center_y1, center_y2 = center_h//4, 3*center_h//4
            center_x1, center_x2 = center_w//4, 3*center_w//4
            
            if center_y2 > center_y1 and center_x2 > center_x1:
                center_roi = roi_depth[center_y1:center_y2, center_x1:center_x2]
                median_depth = np.median(center_roi)
            else:
                median_depth = np.median(roi_depth)
            
            # Convert to real-world distance
            distance_meters = self._depth_to_distance(median_depth)
            
            # Convert to steps
            distance_steps = distance_meters / step_size
            
            return max(0.5, distance_steps)  # Minimum 0.5 steps
            
        except Exception as e:
            logger.error(f"Distance calculation failed: {e}")
            return 2.0
    
    def _depth_to_distance(self, depth_value: float) -> float:
        """Convert MiDaS depth value to real-world distance"""
        try:
            # MiDaS outputs inverse depth, so larger values = closer objects
            # Normalize and invert
            if depth_value <= 0:
                return self.max_distance
            
            # Calibrated conversion (adjust these values based on your setup)
            # This is a simplified model - you may need to calibrate with known distances
            
            # Scale factor based on typical MiDaS output range
            normalized_depth = depth_value / 10.0  # Normalize typical MiDaS range
            
            # Inverse relationship with clamping
            if normalized_depth > 1.0:
                distance = self.min_distance + (2.0 - normalized_depth) * 0.5
            else:
                distance = self.min_distance + (1.0 - normalized_depth) * 5.0
            
            # Clamp to reasonable range
            distance = max(self.min_distance, min(distance, self.max_distance))
            
            return distance
            
        except Exception as e:
            logger.error(f"Depth conversion failed: {e}")
            return 2.0
    
    def calibrate_depth(self, frame: np.ndarray, known_distance: float, 
                       bbox: Tuple[int, int, int, int]):
        """Calibrate depth estimation with known distance"""
        try:
            depth_map = self.estimate_depth(frame)
            x1, y1, x2, y2 = bbox
            
            roi_depth = depth_map[y1:y2, x1:x2]
            measured_depth = np.median(roi_depth)
            
            # Update calibration parameters
            self.baseline_distance = known_distance
            self.baseline_depth_value = measured_depth
            
            logger.info(f"Depth calibrated: {known_distance}m = {measured_depth} depth units")
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
    
    def visualize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """Create visualization of depth map"""
        # Normalize depth map
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        return depth_colored


    def _load_fallback_model(self):
        """Load a fallback depth estimation model"""
        try:
            logger.info("Attempting to load fallback depth model...")
            
            # Try loading the basic MiDaS model
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS", trust_repo=True)
            self.model.to(self.device)
            self.model.eval()
            
            # Load default transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = midas_transforms.default_transform
            
            self.model_type = "MiDaS"
            logger.info("Fallback depth model loaded successfully")
            
        except Exception as e:
            logger.error(f"Fallback model loading failed: {e}")
            # Create a dummy model that returns zeros
            self.model = None
            self.transform = None
            logger.warning("Using dummy depth estimation")