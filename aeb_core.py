#!/usr/bin/env python3
"""
================================================================================
AEB CORE LOGIC MODULE - BAJA SAEINDIA 2025
================================================================================

3-ZONE AEB CONTROLLER:
    Zone 1 (0-100m):     Accelerate to 30 km/h
    Zone 2 (100m+):      Cruise at 30 km/h, detect obstacles
    Zone 3 (braking):    Gradual braking to stop within 6m ± 2m (edge-to-edge)

BAJA SAEINDIA SPEC:
    - Start from rest, accelerate for 100+ meters
    - Detect obstacle and brake
    - Stop within 6m ± 2m (4-8m acceptable range, edge-to-edge)

================================================================================
"""

import math
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    carla = None


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AEBConfig:
    """
    AEB Configuration for BAJA SAEINDIA 2025.
    All distances in meters, speeds in km/h.
    """
    # Zone distances
    zone1_distance: float = 100.0         # Accelerate for first 100m
    
    # Target speed
    target_speed_kmh: float = 30.0
    
    # Braking parameters (edge-to-edge distances)
    # At 30 km/h (8.33 m/s), with ~6 m/s² deceleration, stopping distance ≈ 5.8m
    # Start braking at 12m edge-to-edge to stop at ~6m
    brake_start_distance: float = 12.0    # Start braking at this edge distance
    target_stop_distance: float = 6.0     # Target stop distance (edge-to-edge)
    emergency_distance: float = 4.0       # Full brake below this
    
    # Brake force
    min_brake: float = 0.4                # Initial brake force
    max_brake: float = 1.0                # Maximum brake force
    
    # Detection
    detection_range: float = 200.0
    lane_tolerance: float = 2.5
    
    # Acceleration
    throttle_accel: float = 0.7           # Throttle during acceleration
    throttle_cruise: float = 0.35         # Throttle during cruise
    
    # Stop detection
    stop_speed_threshold: float = 0.3     # Speed below which we consider stopped


# =============================================================================
# ENUMERATIONS
# =============================================================================

class AEBState(Enum):
    ZONE1_ACCELERATING = 1   # First 100m - accelerating
    ZONE2_CRUISING = 2       # After 100m - cruising, looking for obstacles  
    ZONE3_BRAKING = 3        # Obstacle detected - braking
    STOPPED = 4              # Fully stopped


# =============================================================================
# OBSTACLE INFO
# =============================================================================

@dataclass
class ObstacleInfo:
    actor_id: int
    distance: float          # edge-to-edge distance
    lateral_offset: float
    closing_speed: float
    is_in_path: bool


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_speed_kmh(vel) -> float:
    """Get speed in km/h from CARLA velocity."""
    return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6


def calculate_brake_force(distance: float, speed_kmh: float, config: AEBConfig) -> float:
    """
    Calculate brake force based on distance to obstacle.
    Uses physics-based approach for smooth stopping.
    
    At 30 km/h (8.33 m/s):
    - Deceleration needed to stop in X meters: a = v²/(2*X)
    - To stop in 6m: a = 8.33²/(2*6) = 5.8 m/s² (about 0.6g)
    """
    if distance > config.brake_start_distance:
        return 0.0
    
    if distance <= config.emergency_distance:
        return config.max_brake
    
    # Speed in m/s
    speed_ms = speed_kmh / 3.6
    
    # Calculate required deceleration to stop at target distance
    remaining_to_target = distance - config.target_stop_distance
    if remaining_to_target <= 0:
        return config.max_brake
    
    # Required deceleration: a = v²/(2*d)
    required_decel = (speed_ms ** 2) / (2 * remaining_to_target)
    
    # CARLA max deceleration is roughly 8-10 m/s² with full brake
    # Map deceleration to brake force (0.0 to 1.0)
    max_decel = 8.0  # m/s²
    brake = required_decel / max_decel
    
    # Clamp to config limits
    brake = max(config.min_brake, min(config.max_brake, brake))
    
    return brake


# =============================================================================
# OBSTACLE DETECTION
# =============================================================================

def detect_obstacles(ego_vehicle, world, config: AEBConfig) -> List[ObstacleInfo]:
    """
    Detect all obstacles (vehicles, pedestrians, cyclists) ahead.
    Returns edge-to-edge distances.
    """
    obstacles = []
    
    ego_transform = ego_vehicle.get_transform()
    ego_loc = ego_transform.location
    ego_vel = ego_vehicle.get_velocity()
    ego_forward = ego_transform.get_forward_vector()
    
    # Get ego bounding box for edge-to-edge calculation
    ego_bb = ego_vehicle.bounding_box
    ego_half_len = ego_bb.extent.x
    
    # Check vehicles
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.id == ego_vehicle.id:
            continue
        info = _analyze_actor(ego_loc, ego_vel, ego_forward, actor, config, ego_half_len)
        if info:
            obstacles.append(info)
    
    # Check pedestrians
    for actor in world.get_actors().filter('walker.pedestrian.*'):
        info = _analyze_actor(ego_loc, ego_vel, ego_forward, actor, config, ego_half_len)
        if info:
            obstacles.append(info)
    
    return obstacles


def _analyze_actor(ego_loc, ego_vel, ego_forward, actor, config: AEBConfig, ego_half_len: float) -> Optional[ObstacleInfo]:
    """
    Analyze if actor is a collision candidate.
    Returns edge-to-edge distance.
    """
    target_loc = actor.get_location()
    target_vel = actor.get_velocity()
    
    # Get target bounding box
    target_half_len = 0.5  # default for pedestrians
    if hasattr(actor, 'bounding_box'):
        target_half_len = actor.bounding_box.extent.x
    
    # Vector to target
    to_x = target_loc.x - ego_loc.x
    to_y = target_loc.y - ego_loc.y
    
    center_distance = math.sqrt(to_x**2 + to_y**2)
    
    if center_distance > config.detection_range or center_distance < 0.1:
        return None
    
    # Longitudinal distance (along forward vector)
    longitudinal = ego_forward.x * to_x + ego_forward.y * to_y
    
    # Skip if behind
    if longitudinal <= 0:
        return None
    
    # Edge-to-edge distance
    edge_distance = center_distance - ego_half_len - target_half_len
    edge_distance = max(0.0, edge_distance)
    
    # Lateral offset
    lateral = abs(ego_forward.x * to_y - ego_forward.y * to_x)
    
    # Check if in lane
    is_in_lane = lateral < config.lane_tolerance
    
    # Check if crossing into path
    target_speed = math.sqrt(target_vel.x**2 + target_vel.y**2 + target_vel.z**2)
    is_crossing = False
    if target_speed > 0.5 and lateral < 8.0:
        lateral_vel = abs(ego_forward.x * target_vel.y - ego_forward.y * target_vel.x)
        if lateral_vel > target_speed * 0.3:
            is_crossing = True
    
    is_in_path = is_in_lane or is_crossing
    
    # Closing speed
    if center_distance > 0:
        dir_x = to_x / center_distance
        dir_y = to_y / center_distance
        ego_toward = ego_vel.x * dir_x + ego_vel.y * dir_y
        target_toward = target_vel.x * dir_x + target_vel.y * dir_y
        closing_speed = ego_toward - target_toward
    else:
        closing_speed = 0.0
    
    return ObstacleInfo(
        actor_id=actor.id,
        distance=edge_distance,
        lateral_offset=lateral,
        closing_speed=closing_speed,
        is_in_path=is_in_path
    )


def find_critical_obstacle(obstacles: List[ObstacleInfo]) -> Optional[ObstacleInfo]:
    """Find closest obstacle in collision path."""
    in_path = [o for o in obstacles if o.is_in_path]
    if not in_path:
        return None
    in_path.sort(key=lambda o: o.distance)
    return in_path[0]


# =============================================================================
# AEB CONTROLLER - 3 ZONE LOGIC
# =============================================================================

class AEBController:
    """
    3-Zone AEB Controller for BAJA SAEINDIA 2025.
    
    Zone 1 (0-100m):   Accelerate to target speed
    Zone 2 (100m+):    Cruise and detect obstacles
    Zone 3 (braking):  Gradual braking to stop within 6m ± 2m
    """
    
    def __init__(self, config: AEBConfig = None):
        self.config = config or AEBConfig()
        self._state = AEBState.ZONE1_ACCELERATING
        self._distance_traveled = 0.0
        self._last_pos = None
        self._stopped = False
        self._last_obstacle_distance = None
        self._last_ttc = None
    
    def reset(self):
        """Reset for new scenario."""
        self._state = AEBState.ZONE1_ACCELERATING
        self._distance_traveled = 0.0
        self._last_pos = None
        self._stopped = False
        self._last_obstacle_distance = None
        self._last_ttc = None
    
    @property
    def state(self) -> AEBState:
        return self._state
    
    @property
    def distance_traveled(self) -> float:
        return self._distance_traveled
    
    @property
    def is_stopped(self) -> bool:
        return self._stopped
    
    def get_state(self) -> dict:
        """Get current state as dictionary for scenario scripts."""
        return {
            'state': self._state.name,
            'distance_traveled': self._distance_traveled,
            'stopped': self._stopped,
            'obstacle_distance': self._last_obstacle_distance,
            'ttc': self._last_ttc
        }
    
    def update(self, ego_vehicle, world) -> 'carla.VehicleControl':
        """
        Main update - call every tick.
        Returns CARLA VehicleControl with throttle/brake.
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        
        # Get ego state
        ego_loc = ego_vehicle.get_location()
        ego_vel = ego_vehicle.get_velocity()
        speed_kmh = get_speed_kmh(ego_vel)
        
        # Update distance traveled
        if self._last_pos:
            dx = ego_loc.x - self._last_pos[0]
            dy = ego_loc.y - self._last_pos[1]
            self._distance_traveled += math.sqrt(dx**2 + dy**2)
        self._last_pos = (ego_loc.x, ego_loc.y)
        
        # Already stopped?
        if self._stopped:
            control.throttle = 0.0
            control.brake = 1.0
            return control
        
        # Only detect obstacles AFTER Zone 1 (ego doesn't "know" obstacle location during acceleration)
        # This simulates realistic sensor behavior - detection starts after 100m traveled
        critical = None
        if self._distance_traveled >= self.config.zone1_distance:
            obstacles = detect_obstacles(ego_vehicle, world, self.config)
            critical = find_critical_obstacle(obstacles)
        
        # Store for get_state() - only show obstacle info after Zone 1
        if critical:
            self._last_obstacle_distance = critical.distance
            if critical.closing_speed > 0.1:
                self._last_ttc = critical.distance / critical.closing_speed
            else:
                self._last_ttc = None
        else:
            self._last_obstacle_distance = None
            self._last_ttc = None
        
        # =================================================================
        # 3-ZONE DECISION LOGIC
        # =================================================================
        
        # Check if we need to brake (overrides zone logic)
        if critical and critical.distance <= self.config.brake_start_distance:
            # ZONE 3: BRAKING
            self._state = AEBState.ZONE3_BRAKING
            brake = calculate_brake_force(critical.distance, speed_kmh, self.config)
            control.throttle = 0.0
            control.brake = brake
            
            # Check if stopped
            if speed_kmh < self.config.stop_speed_threshold:
                self._stopped = True
                self._state = AEBState.STOPPED
                control.brake = 1.0
        
        elif self._distance_traveled < self.config.zone1_distance:
            # ZONE 1: ACCELERATING (first 100m)
            self._state = AEBState.ZONE1_ACCELERATING
            
            if speed_kmh < self.config.target_speed_kmh:
                control.throttle = self.config.throttle_accel
            else:
                control.throttle = self.config.throttle_cruise
            control.brake = 0.0
        
        else:
            # ZONE 2: CRUISING (after 100m, no obstacle in braking range)
            self._state = AEBState.ZONE2_CRUISING
            
            if speed_kmh < self.config.target_speed_kmh - 2:
                control.throttle = self.config.throttle_accel
            else:
                control.throttle = self.config.throttle_cruise
            control.brake = 0.0
        
        return control


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_aeb_controller(
    zone1_distance: float = 100.0,
    target_speed_kmh: float = 30.0,
    brake_start_distance: float = 12.0,
    target_stop_distance: float = 6.0
) -> AEBController:
    """Create AEB controller with custom parameters."""
    config = AEBConfig(
        zone1_distance=zone1_distance,
        target_speed_kmh=target_speed_kmh,
        brake_start_distance=brake_start_distance,
        target_stop_distance=target_stop_distance
    )
    return AEBController(config)
