import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import time

@dataclass
class TrackedObject:
    class_name: str
    position: Tuple[float, float]
    distance: float
    direction: str
    confidence: float
    first_seen: float
    last_seen: float
    stable_count: int = 0
    announced: bool = False  # Track if object has been announced

class SceneTracker:
    def __init__(self):
        """Initialize scene tracker"""
        self.tracked_objects: Dict[str, TrackedObject] = {}
        self.last_announcement_time = 0
        self.min_announcement_interval = 3.0  # Increased interval
        
        # Thresholds for detecting significant changes
        self.position_threshold = 80  # Increased threshold (less sensitive)
        self.distance_threshold = 1.5  # Increased threshold
        self.direction_change_threshold = 45  # Increased threshold
        self.stability_frames = 5  # More frames for stability
        
        # Object persistence
        self.disappear_timeout = 3.0  # Seconds before object is considered gone
        
        logger.info("Scene tracker initialized")
    
    def update_scene(self, detected_objects: List) -> Dict[str, str]:
        """Update tracked scene and return meaningful changes"""
        current_time = time.time()
        changes = {}
        
        # Create current object map with better grouping
        current_objects = {}
        for obj in detected_objects:
            # Group objects by class and approximate position
            region_x = obj.center_point[0] // 100  # Larger regions
            region_y = obj.center_point[1] // 100
            key = f"{obj.class_name}_{region_x}_{region_y}"
            
            # If multiple objects of same type in region, keep the most confident one
            if key not in current_objects or obj.confidence > current_objects[key].confidence:
                current_objects[key] = obj
        
        # Update existing objects and detect meaningful changes
        for key, obj in current_objects.items():
            if key not in self.tracked_objects:
                # New object - but only announce if it's stable
                self.tracked_objects[key] = TrackedObject(
                    class_name=obj.class_name,
                    position=obj.center_point,
                    distance=obj.distance if obj.distance else 2.0,
                    direction=obj.direction,
                    confidence=obj.confidence,
                    first_seen=current_time,
                    last_seen=current_time,
                    announced=False
                )
                
            else:
                # Update existing object
                tracked = self.tracked_objects[key]
                tracked.last_seen = current_time
                
                # Check for significant changes
                pos_change = self._calculate_position_change(tracked.position, obj.center_point)
                distance_change = abs(tracked.distance - (obj.distance or 2.0))
                direction_changed = tracked.direction != obj.direction
                
                # Only announce meaningful changes
                if tracked.announced:  # Only track changes for already announced objects
                    change_description = []
                    
                    if distance_change > self.distance_threshold:
                        if obj.distance and obj.distance < tracked.distance - 0.5:
                            change_description.append("getting closer")
                        elif obj.distance and obj.distance > tracked.distance + 0.5:
                            change_description.append("moving away")
                        tracked.distance = obj.distance or 2.0
                    
                    if direction_changed and pos_change > self.position_threshold:
                        change_description.append(f"moved to your {obj.direction}")
                        tracked.direction = obj.direction
                        tracked.position = obj.center_point
                    
                    if change_description:
                        tracked.stable_count = 0
                        changes[f"change_{key}"] = self._format_object_change(obj, change_description)
                else:
                    # Check if object is stable enough to announce
                    if pos_change <= self.position_threshold:
                        tracked.stable_count += 1
                        if tracked.stable_count >= self.stability_frames:
                            tracked.announced = True
                            changes[f"stable_{key}"] = self._format_stable_object(obj)
                    else:
                        tracked.stable_count = 0
                        tracked.position = obj.center_point
        
        # Check for objects that have been gone for a while
        disappeared_keys = []
        for key, tracked in self.tracked_objects.items():
            if key not in current_objects:
                if current_time - tracked.last_seen > self.disappear_timeout:
                    if tracked.announced:  # Only announce disappearance for previously announced objects
                        changes[f"gone_{key}"] = self._format_disappeared_object(tracked)
                    disappeared_keys.append(key)
        
        # Remove disappeared objects
        for key in disappeared_keys:
            del self.tracked_objects[key]
        
        return changes
    
    def _calculate_position_change(self, old_pos: Tuple[float, float], 
                                 new_pos: Tuple[float, float]) -> float:
        """Calculate distance between two positions"""
        return np.sqrt((old_pos[0] - new_pos[0])**2 + (old_pos[1] - new_pos[1])**2)
    
    def _format_stable_object(self, obj) -> str:
        """Format announcement for stable new object"""
        distance_text = f" {obj.distance:.0f} steps away" if obj.distance and obj.distance > 0 else ""
        direction_text = f" {self._format_direction(obj.direction)}" if obj.direction else ""
        return f"There's a {obj.class_name}{distance_text}{direction_text}"
    
    def _format_object_change(self, obj, changes: List[str]) -> str:
        """Format announcement for object change"""
        change_text = " and ".join(changes)
        return f"The {obj.class_name} is {change_text}"
    
    def _format_disappeared_object(self, tracked: TrackedObject) -> str:
        """Format announcement for disappeared object"""
        return f"The {tracked.class_name} has moved away"
    
    def _format_direction(self, direction: str) -> str:
        """Format direction text"""
        if direction == "center":
            return "ahead"
        elif direction == "left":
            return "to your left"
        elif direction == "right":
            return "to your right"
        else:
            return ""
    
    def should_announce(self) -> bool:
        """Check if enough time has passed for new announcement"""
        current_time = time.time()
        if current_time - self.last_announcement_time >= self.min_announcement_interval:
            self.last_announcement_time = current_time
            return True
        return False
    
    def get_scene_summary(self) -> str:
        """Get summary of current stable scene"""
        if not self.tracked_objects:
            return "I don't see anything specific right now"
        
        # Only include announced (stable) objects
        stable_objects = [
            obj for obj in self.tracked_objects.values() 
            if obj.announced and obj.stable_count >= self.stability_frames
        ]
        
        if not stable_objects:
            return "I'm still analyzing the scene"
        
        # Group by class name
        object_counts = {}
        closest_objects = {}
        
        for obj in stable_objects:
            class_name = obj.class_name
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
            
            if class_name not in closest_objects or obj.distance < closest_objects[class_name].distance:
                closest_objects[class_name] = obj
        
        # Format summary
        items = []
        for class_name, count in object_counts.items():
            obj = closest_objects[class_name]
            distance_text = f" {obj.distance:.0f} steps away" if obj.distance > 0 else ""
            
            if count == 1:
                items.append(f"a {class_name}{distance_text}")
            else:
                items.append(f"{count} {class_name}s")
        
        if not items:
            return "I'm still analyzing the scene"
        elif len(items) == 1:
            return f"I can see {items[0]}"
        elif len(items) == 2:
            return f"I can see {items[0]} and {items[1]}"
        else:
            return f"I can see {', '.join(items[:-1])}, and {items[-1]}"
    
    def reset(self):
        """Reset tracker state"""
        self.tracked_objects.clear()
        self.last_announcement_time = 0
        logger.info("Scene tracker reset")
