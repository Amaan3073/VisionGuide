import cv2
import torch
import numpy as np
from typing import Tuple, Optional
import urllib.request
import os
from loguru import logger

class DepthEstimator:
    def __init__(self, model_type: str = "MiDaS_small"):
        """
        Initialize depth estimation model
        
        Args:
            model_type: Type of MiDaS model to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model = None
        self.transform = None
        self._load_model()
        
    def _load_model(self):
        """Load MiDaS depth estimation model"""
        try:
            # Updated model names for current MiDaS repository
            valid_models = {
                "MiDaS_small": "MiDaS_small",
                "MiDaS": "MiDaS",
                "DPT_Large": "DPT_Large",
                "DPT_Hybrid": "DPT_Hybrid",
                "DPT_SwinV2_L_384": "DPT_SwinV2_L_384",
                "DPT_SwinV2_B_384": "DPT_SwinV2_B_384",
                "DPT_SwinV2_T_256": "DPT_SwinV2_T_256",
                "DPT_BEiT_L_384": "DPT_BEiT_L_384",
                "DPT_BEiT_B_384": "DPT_BEiT_B_384",
                "DPT_BEiT_L_512": "DPT_BEiT_L_512",
                "DPT_Levit_224": "DPT_Levit_224",
                "DPT_Next_ViT_L_384": "DPT_Next_ViT_L_384"
            }
            
            # Use default model if specified model not found
            if self.model_type not in valid_models:
                logger.warning(f"Model {self.model_type} not found, using MiDaS_small")
                self.model_type = "MiDaS_small"
            
            # Load MiDaS model
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
            self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            
            # Select appropriate transform based on model type
            if "small" in self.model_type.lower():
                self.transform = midas_transforms.small_transform
            elif "DPT" in self.model_type:
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.default_transform
                
            logger.info(f"Depth estimation model loaded: {self.model_type}")
            
        except Exception as e:
            logger.error(f"Failed to load depth model: {e}")
            # Fallback to a simpler approach
            self._load_fallback_model()
    
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
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from input frame
        
        Args:
            frame: Input RGB frame
            
        Returns:
            Depth map as numpy array
        """
        if self.model is None or self.transform is None:
            # Return dummy depth map
            return np.ones(frame.shape[:2]) * 5.0
        
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
            return np.ones(frame.shape[:2]) * 5.0
    
    # ... rest of the methods remain the same

    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from input frame
        
        Args:
            frame: Input RGB frame
            
        Returns:
            Depth map as numpy array
        """
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
            return np.zeros(frame.shape[:2])
    
    def get_object_distance(self, depth_map: np.ndarray, 
                           bbox: Tuple[int, int, int, int],
                           step_size: float = 0.75) -> float:
        """
        Calculate object distance from depth map
        
        Args:
            depth_map: Depth map from estimate_depth
            bbox: Bounding box (x1, y1, x2, y2)
            step_size: User's step size in meters
            
        Returns:
            Distance in steps
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Extract depth values in bounding box
            roi_depth = depth_map[y1:y2, x1:x2]
            
            # Calculate median depth (more robust than mean)
            median_depth = np.median(roi_depth)
            
            # Convert relative depth to approximate distance
            # This is a simplified conversion - you may need calibration
            distance_meters = self._depth_to_distance(median_depth)
            
            # Convert to steps
            distance_steps = distance_meters / step_size
            
            return distance_steps
            
        except Exception as e:
            logger.error(f"Distance calculation failed: {e}")
            return 0.0
    
    def _depth_to_distance(self, depth_value: float) -> float:
        """
        Convert MiDaS depth value to real-world distance
        
        Note: This is a simplified conversion. For accurate results,
        you would need to calibrate with known distances.
        
        Args:
            depth_value: MiDaS depth value
            
        Returns:
            Approximate distance in meters
        """
        # Simple inverse relationship (needs calibration)
        # Smaller depth values = closer objects
        if depth_value <= 0:
            return 10.0  # Default far distance
        
        # Empirical conversion (adjust based on testing)
        distance = 10.0 / (depth_value + 0.1)
        
        # Clamp to reasonable range
        distance = max(0.5, min(distance, 20.0))
        
        return distance
    
    def visualize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Create visualization of depth map
        
        Args:
            depth_map: Depth map from estimate_depth
            
        Returns:
            Colorized depth map
        """
        # Normalize depth map
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        return depth_colored
