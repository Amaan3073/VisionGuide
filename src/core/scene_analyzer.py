import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter
from loguru import logger

@dataclass
class SceneContext:
    environment_type: str  # indoor, outdoor, kitchen, etc.
    dominant_objects: List[str]
    spatial_relationships: List[str]
    safety_alerts: List[str]
    summary: str

class SceneAnalyzer:
    def __init__(self):
        """Initialize scene analyzer with object categories"""
        self.indoor_objects = {
            'chair', 'couch', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'keyboard', 'mouse', 'book', 'clock', 'scissors', 'hair drier',
            'toothbrush', 'sink', 'refrigerator', 'microwave', 'oven', 'toaster'
        }
        
        self.outdoor_objects = {
            'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe'
        }
        
        self.kitchen_objects = {
            'refrigerator', 'microwave', 'oven', 'toaster', 'sink', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
            'pizza', 'donut', 'cake'
        }
        
        self.safety_hazards = {
            'stairs', 'fire hydrant', 'stop sign', 'traffic light', 'car',
            'truck', 'bus', 'motorcycle', 'bicycle'
        }
    
    def analyze_scene(self, detected_objects: List, 
                     depth_map: np.ndarray = None) -> SceneContext:
        """
        Analyze scene from detected objects
        
        Args:
            detected_objects: List of DetectedObject instances
            depth_map: Optional depth map for spatial analysis
            
        Returns:
            SceneContext with analysis results
        """
        try:
            # Extract object names
            object_names = [obj.class_name for obj in detected_objects]
            
            # Determine environment type
            environment_type = self._classify_environment(object_names)
            
            # Find dominant objects
            dominant_objects = self._get_dominant_objects(object_names)
            
            # Analyze spatial relationships
            spatial_relationships = self._analyze_spatial_relationships(detected_objects)
            
            # Check for safety hazards
            safety_alerts = self._check_safety_hazards(detected_objects)
            
            # Generate summary
            summary = self._generate_summary(
                environment_type, dominant_objects, 
                spatial_relationships, safety_alerts
            )
            
            return SceneContext(
                environment_type=environment_type,
                dominant_objects=dominant_objects,
                spatial_relationships=spatial_relationships,
                safety_alerts=safety_alerts,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Scene analysis failed: {e}")
            return SceneContext(
                environment_type="unknown",
                dominant_objects=[],
                spatial_relationships=[],
                safety_alerts=[],
                summary="Unable to analyze scene"
            )
    
    def _classify_environment(self, object_names: List[str]) -> str:
        """Classify environment type based on objects"""
        indoor_count = len(set(object_names) & self.indoor_objects)
        outdoor_count = len(set(object_names) & self.outdoor_objects)
        kitchen_count = len(set(object_names) & self.kitchen_objects)
        
        if kitchen_count >= 2:
            return "kitchen"
        elif indoor_count > outdoor_count:
            return "indoor"
        elif outdoor_count > 0:
            return "outdoor"
        else:
            return "unknown"
    
    def _get_dominant_objects(self, object_names: List[str], 
                            max_objects: int = 5) -> List[str]:
        """Get most frequent objects in scene"""
        counter = Counter(object_names)
        return [obj for obj, count in counter.most_common(max_objects)]
    
    def _analyze_spatial_relationships(self, detected_objects: List) -> List[str]:
        """Analyze spatial relationships between objects"""
        relationships = []
        
        # Sort objects by distance if available
        objects_with_distance = [obj for obj in detected_objects if obj.distance]
        if objects_with_distance:
            objects_with_distance.sort(key=lambda x: x.distance)
            
            # Closest object
            closest = objects_with_distance[0]
            relationships.append(f"{closest.class_name} is closest at {closest.distance:.1f} steps")
            
            # Furthest object
            if len(objects_with_distance) > 1:
                furthest = objects_with_distance[-1]
                relationships.append(f"{furthest.class_name} is furthest at {furthest.distance:.1f} steps")
        
        # Analyze directions
        left_objects = [obj for obj in detected_objects if obj.direction == "left"]
        right_objects = [obj for obj in detected_objects if obj.direction == "right"]
        center_objects = [obj for obj in detected_objects if obj.direction == "center"]
        
        if left_objects:
            relationships.append(f"On your left: {', '.join([obj.class_name for obj in left_objects[:3]])}")
        if right_objects:
            relationships.append(f"On your right: {', '.join([obj.class_name for obj in right_objects[:3]])}")
        if center_objects:
            relationships.append(f"Ahead: {', '.join([obj.class_name for obj in center_objects[:3]])}")
        
        return relationships
    
    def _check_safety_hazards(self, detected_objects: List) -> List[str]:
        """Check for potential safety hazards"""
        hazards = []
        
        for obj in detected_objects:
            if obj.class_name in self.safety_hazards:
                distance_text = f" {obj.distance:.1f} steps away" if obj.distance else ""
                direction_text = f" on your {obj.direction}" if obj.direction != "center" else " ahead"
                hazards.append(f"Warning: {obj.class_name}{direction_text}{distance_text}")
        
        return hazards
    
    def _generate_summary(self, environment_type: str, 
                         dominant_objects: List[str],
                         spatial_relationships: List[str],
                         safety_alerts: List[str]) -> str:
        """Generate natural language summary of scene"""
        summary_parts = []
        
        # Environment
        if environment_type != "unknown":
            summary_parts.append(f"You are in a {environment_type} environment.")
        
        # Dominant objects
        if dominant_objects:
            if len(dominant_objects) == 1:
                summary_parts.append(f"I can see a {dominant_objects[0]}.")
            else:
                objects_text = ", ".join(dominant_objects[:-1]) + f" and {dominant_objects[-1]}"
                summary_parts.append(f"I can see {objects_text}.")
        
        # Spatial information
        if spatial_relationships:
            summary_parts.extend(spatial_relationships[:2])  # Limit to 2 relationships
        
        # Safety alerts
        if safety_alerts:
            summary_parts.extend(safety_alerts)
        
        return " ".join(summary_parts)

def format_for_audio(scene_context: SceneContext, 
                    detected_objects: List,
                    max_objects: int = 3) -> str:
    """
    Format scene analysis for audio output
    
    Args:
        scene_context: Analyzed scene context
        detected_objects: List of detected objects
        max_objects: Maximum objects to mention
        
    Returns:
        Formatted string for TTS
    """
    audio_parts = []
    
    # Priority: Safety alerts first
    if scene_context.safety_alerts:
        audio_parts.extend(scene_context.safety_alerts[:2])
    
    # Environment and objects
    if scene_context.summary:
        audio_parts.append(scene_context.summary)
    else:
        # Fallback: mention closest objects
        close_objects = [obj for obj in detected_objects if obj.distance and obj.distance <= 5]
        if close_objects:
            close_objects.sort(key=lambda x: x.distance)
            for obj in close_objects[:max_objects]:
                distance_text = f"{obj.distance:.0f} steps away"
                direction_text = f"on your {obj.direction}" if obj.direction != "center" else "ahead"
                audio_parts.append(f"{obj.class_name} {distance_text} {direction_text}")
    
    return ". ".join(audio_parts) + "."
