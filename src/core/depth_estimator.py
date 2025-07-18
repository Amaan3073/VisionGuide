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
        
        # Enhanced calibration parameters
        self.focal_length = 525.0  # Approximate focal length for webcam
        self.baseline = 0.1  # Baseline in meters
        self.depth_scale = 1000.0  # Scale factor for depth values
        
        # Distance mapping parameters
        self.min_distance = 0.3  # meters
        self.max_distance = 15.0  # meters
        self.step_size = 0.75  # meters per step
        
        # Calibration lookup table for better accuracy
        self.distance_map = {
            # depth_value_range: actual_distance_meters
            (0, 0.1): 0.5,
            (0.1, 0.2): 1.0,
            (0.2, 0.3): 1.5,
            (0.3, 0.4): 2.0,
            (0.4, 0.5): 2.5,
            (0.5, 0.6): 3.0,
            (0.6, 0.7): 4.0,
            (0.7, 0.8): 5.0,
            (0.8, 0.9): 7.0,
            (0.9, 1.0): 10.0,
        }
        
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
            return np.ones(frame.shape[:2]) * 0.5  # Return default depth
        
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
            
            # Convert to numpy and normalize
            depth_map = prediction.cpu().numpy()
            
            # Normalize depth map to 0-1 range
            depth_map = self._normalize_depth(depth_map)
            
            return depth_map
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return np.ones(frame.shape[:2]) * 0.5
    
    def _normalize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """Normalize depth map to 0-1 range"""
        # Remove outliers
        depth_map = np.clip(depth_map, 
                           np.percentile(depth_map, 2), 
                           np.percentile(depth_map, 98))
        
        # Normalize to 0-1
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)
        
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_map = np.ones_like(depth_map) * 0.5
        
        return depth_map
    
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
                return 3.0  # Default distance in steps
            
            # Extract depth values in bounding box
            roi_depth = depth_map[y1:y2, x1:x2]
            
            # Use center region for more stable depth estimation
            center_h, center_w = roi_depth.shape
            pad_h, pad_w = center_h // 4, center_w // 4
            
            if center_h > 2 * pad_h and center_w > 2 * pad_w:
                center_roi = roi_depth[pad_h:center_h-pad_h, pad_w:center_w-pad_w]
                if center_roi.size > 0:
                    # Use 25th percentile for more stable reading (excludes outliers)
                    depth_value = np.percentile(center_roi, 25)
                else:
                    depth_value = np.median(roi_depth)
            else:
                depth_value = np.median(roi_depth)
            
            # Convert to real-world distance
            distance_meters = self._depth_to_distance(depth_value)
            
            # Convert to steps
            distance_steps = distance_meters / step_size
            
            # Round to reasonable precision
            distance_steps = round(distance_steps, 1)
            
            return max(0.5, distance_steps)  # Minimum 0.5 steps
            
        except Exception as e:
            logger.error(f"Distance calculation failed: {e}")
            return 3.0
    
    def _depth_to_distance(self, depth_value: float) -> float:
        """Convert normalized depth value to real-world distance"""
        try:
            # Clamp depth value to valid range
            depth_value = max(0.0, min(1.0, depth_value))
            
            # Use lookup table for better accuracy
            for (min_depth, max_depth), distance in self.distance_map.items():
                if min_depth <= depth_value < max_depth:
                    return distance
            
            # Fallback: inverse relationship with scaling
            if depth_value < 0.05:
                return self.min_distance
            elif depth_value > 0.95:
                return self.max_distance
            else:
                # Logarithmic mapping for better distribution
                # Smaller depth values (closer to 0) = closer objects
                # Larger depth values (closer to 1) = farther objects
                distance = self.min_distance + (1 - depth_value) * (self.max_distance - self.min_distance)
                
                # Apply logarithmic scaling for more realistic distances
                distance = self.min_distance + (distance - self.min_distance) ** 1.5
                
                return min(self.max_distance, max(self.min_distance, distance))
            
        except Exception as e:
            logger.error(f"Depth conversion failed: {e}")
            return 2.0
    
    def get_distance_category(self, distance_steps: float) -> str:
        """Get distance category for better user feedback"""
        if distance_steps < 1.0:
            return "very close"
        elif distance_steps < 2.0:
            return "close"
        elif distance_steps < 4.0:
            return "nearby"
        elif distance_steps < 8.0:
            return "moderate distance"
        else:
            return "far away"
    
    def calibrate_with_known_object(self, frame: np.ndarray, 
                                   bbox: Tuple[int, int, int, int],
                                   known_distance: float):
        """Calibrate depth estimation with a known object distance"""
        try:
            depth_map = self.estimate_depth(frame)
            x1, y1, x2, y2 = bbox
            
            roi_depth = depth_map[y1:y2, x1:x2]
            measured_depth = np.median(roi_depth)
            
            # Update calibration
            logger.info(f"Calibration: {known_distance}m object has depth value {measured_depth:.3f}")
            
            # You can use this information to adjust the distance mapping
            return measured_depth
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return None
    
    def visualize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """Create visualization of depth map"""
        # Normalize depth map for visualization
        depth_normalized = (depth_map * 255).astype(np.uint8)
        
        # Apply colormap (closer = red, farther = blue)
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