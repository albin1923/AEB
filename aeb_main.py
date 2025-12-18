#!/usr/bin/env python3
"""
AEB Controller for BAJA SAEINDIA Competition
============================================
Specification:
- Zone 1: 45 meters - Accelerate to 30 km/h (+/- 3 km/h)
- Zone 2: Detect stationary obstacle and apply brakes
- Zone 3: Stop within 6 meters (+/- 3m) from obstacle

Clean implementation for C++ portability.
"""

import carla
import pygame
import numpy as np
import json
import time
import math
import random

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Connection
    'host': 'localhost',
    'port': 2000,
    'timeout': 10.0,
    
    # Simulation
    'fixed_delta': 0.05,  # 20 FPS physics
    
    # Zone 1 - Acceleration Phase
    'zone1_distance': 45.0,  # meters to accelerate
    'target_speed_kmh': 30.0,  # target speed in km/h
    'speed_tolerance_kmh': 3.0,  # +/- tolerance
    
    # Zone 3 - Stopping Phase
    'target_stop_distance': 6.0,  # target stop distance from obstacle (3-9m acceptable)
    
    # Braking Parameters - SIMPLE LINEAR APPROACH
    'brake_start_distance': 14.0,  # start braking when obstacle within this distance
    'min_brake': 0.4,  # minimum brake force
    'max_brake': 1.0,  # maximum brake force
    
    # Detection
    'detection_range': 100.0,  # max detection range in meters
    
    # Spawner sync file
    'spawn_info_file': '/tmp/aeb_spawn_info.json',
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_speed_kmh(vehicle):
    """Get vehicle speed in km/h."""
    vel = vehicle.get_velocity()
    speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    return speed_ms * 3.6


def get_distance_to_vehicle(ego, target):
    """Get Euclidean distance between two vehicles."""
    ego_loc = ego.get_location()
    target_loc = target.get_location()
    return math.sqrt(
        (ego_loc.x - target_loc.x)**2 +
        (ego_loc.y - target_loc.y)**2
    )


def get_distance_traveled(start_location, current_location):
    """Get distance traveled from start."""
    return math.sqrt(
        (current_location.x - start_location.x)**2 +
        (current_location.y - start_location.y)**2
    )


def find_obstacle_ahead(ego_vehicle, world, lane_tolerance=2.0):
    """Find vehicle in front within same lane."""
    ego_loc = ego_vehicle.get_location()
    ego_transform = ego_vehicle.get_transform()
    ego_forward = ego_transform.get_forward_vector()
    
    all_vehicles = world.get_actors().filter('vehicle.*')
    
    closest_distance = float('inf')
    closest_vehicle = None
    
    for vehicle in all_vehicles:
        if vehicle.id == ego_vehicle.id:
            continue
            
        target_loc = vehicle.get_location()
        
        # Vector from ego to target
        to_target = carla.Vector3D(
            target_loc.x - ego_loc.x,
            target_loc.y - ego_loc.y,
            0
        )
        
        # Check if target is ahead (dot product positive)
        dot = ego_forward.x * to_target.x + ego_forward.y * to_target.y
        if dot <= 0:
            continue  # Vehicle is behind us
        
        # Calculate lateral offset
        distance = math.sqrt(to_target.x**2 + to_target.y**2)
        if distance == 0:
            continue
            
        # Normalize to_target
        to_target_norm = carla.Vector3D(
            to_target.x / distance,
            to_target.y / distance,
            0
        )
        
        # Cross product gives lateral offset
        cross = ego_forward.x * to_target_norm.y - ego_forward.y * to_target_norm.x
        lateral_offset = abs(cross * distance)
        
        if lateral_offset > lane_tolerance:
            continue  # Vehicle is in different lane
        
        if distance < closest_distance:
            closest_distance = distance
            closest_vehicle = vehicle
    
    return closest_vehicle, closest_distance if closest_vehicle else None


def calculate_brake_force(distance_to_obstacle, current_speed_kmh):
    """
    Simple linear braking: closer = harder braking.
    
    This is designed for C++ portability - no complex physics.
    """
    if distance_to_obstacle > CONFIG['brake_start_distance']:
        return 0.0
    
    # Linear interpolation: far = min_brake, close = max_brake
    # At brake_start_distance: min_brake
    # At target_stop_distance: max_brake
    
    range_distance = CONFIG['brake_start_distance'] - CONFIG['target_stop_distance']
    current_from_target = distance_to_obstacle - CONFIG['target_stop_distance']
    
    if current_from_target <= 0:
        return CONFIG['max_brake']
    
    # Normalized position (1.0 = at brake_start, 0.0 = at target_stop)
    normalized = current_from_target / range_distance
    normalized = max(0.0, min(1.0, normalized))
    
    # Invert: closer = higher brake
    brake = CONFIG['min_brake'] + (1.0 - normalized) * (CONFIG['max_brake'] - CONFIG['min_brake'])
    
    # Speed-based adjustment: faster = more brake needed
    speed_factor = current_speed_kmh / CONFIG['target_speed_kmh']
    brake *= max(1.0, speed_factor)
    
    return min(CONFIG['max_brake'], brake)


def find_all_straight_spawns(world, min_length=150.0):
    """Find spawn points on straight road segments."""
    spawn_points = world.get_map().get_spawn_points()
    carla_map = world.get_map()
    straight_spawns = []
    
    for sp in spawn_points:
        waypoint = carla_map.get_waypoint(sp.location)
        if waypoint is None:
            continue
        
        # Check road curvature ahead
        is_straight = True
        initial_yaw = waypoint.transform.rotation.yaw
        check_wp = waypoint
        
        for _ in range(int(min_length / 5)):
            next_wps = check_wp.next(5.0)
            if not next_wps:
                is_straight = False
                break
            check_wp = next_wps[0]
            
            yaw_diff = abs(check_wp.transform.rotation.yaw - initial_yaw)
            if yaw_diff > 180:
                yaw_diff = 360 - yaw_diff
            if yaw_diff > 5:  # Allow max 5 degree deviation
                is_straight = False
                break
        
        if is_straight:
            straight_spawns.append(sp)
    
    return straight_spawns


def clear_lane_ahead(world, spawn_point, distance=120.0):
    """Remove any vehicles in the lane ahead of spawn point."""
    carla_map = world.get_map()
    waypoint = carla_map.get_waypoint(spawn_point.location)
    
    if waypoint is None:
        return
    
    # Get all waypoints in lane ahead
    lane_waypoints = [waypoint]
    current = waypoint
    for _ in range(int(distance / 3)):
        next_wps = current.next(3.0)
        if not next_wps:
            break
        current = next_wps[0]
        lane_waypoints.append(current)
    
    # Remove vehicles near these waypoints
    all_vehicles = world.get_actors().filter('vehicle.*')
    for vehicle in all_vehicles:
        v_loc = vehicle.get_location()
        for wp in lane_waypoints:
            if v_loc.distance(wp.transform.location) < 4.0:
                vehicle.destroy()
                break


def write_spawn_info(spawn_point, road_id, lane_id):
    """Write spawn info for obstacle spawner."""
    info = {
        'x': spawn_point.location.x,
        'y': spawn_point.location.y,
        'z': spawn_point.location.z,
        'yaw': spawn_point.rotation.yaw,
        'road_id': road_id,
        'lane_id': lane_id,
        'timestamp': time.time(),
        'ready': True
    }
    with open(CONFIG['spawn_info_file'], 'w') as f:
        json.dump(info, f)


# =============================================================================
# MAIN AEB CONTROLLER
# =============================================================================

class AEBController:
    def __init__(self):
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.camera = None
        self.display = None
        
        # State tracking
        self.start_location = None
        self.zone1_complete = False
        self.obstacle_detected = False
        self.test_complete = False
        self.final_distance = None
        self.image_data = None  # Buffer for camera image
        
    def connect(self):
        """Connect to CARLA server."""
        print(f"Connecting to CARLA at {CONFIG['host']}:{CONFIG['port']}...")
        self.client = carla.Client(CONFIG['host'], CONFIG['port'])
        self.client.set_timeout(CONFIG['timeout'])
        self.world = self.client.get_world()
        
        # Enable synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = CONFIG['fixed_delta']
        self.world.apply_settings(settings)
        
        print("Connected to CARLA")
        
    def spawn_ego_vehicle(self):
        """Spawn ego vehicle on straight road."""
        straight_spawns = find_all_straight_spawns(self.world)
        
        if not straight_spawns:
            print("ERROR: No straight road segments found")
            return False
        
        # Pick random spawn
        spawn_point = random.choice(straight_spawns)
        
        # Clear lane ahead
        clear_lane_ahead(self.world, spawn_point)
        self.world.tick()
        
        # Get road info for spawner
        carla_map = self.world.get_map()
        waypoint = carla_map.get_waypoint(spawn_point.location)
        
        # Write spawn info for obstacle spawner
        write_spawn_info(spawn_point, waypoint.road_id, waypoint.lane_id)
        
        # Spawn Tesla Model 3
        blueprint_lib = self.world.get_blueprint_library()
        vehicle_bp = blueprint_lib.find('vehicle.tesla.model3')
        
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.start_location = spawn_point.location
        
        print(f"Spawned ego vehicle at ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
        print(f"Road ID: {waypoint.road_id}, Lane ID: {waypoint.lane_id}")
        
        return True
        
    def setup_camera(self):
        """Setup back POV camera for visualization."""
        pygame.init()
        self.display = pygame.display.set_mode((800, 600))
        pygame.display.set_caption('AEB Controller - BAJA SAEINDIA')
        
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        
        # Back POV camera position
        camera_transform = carla.Transform(
            carla.Location(x=-8, z=4),
            carla.Rotation(pitch=-15)
        )
        
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.ego_vehicle
        )
        
        # Store image in buffer instead of rendering directly
        self.camera.listen(lambda image: self._store_image(image))
    
    def _store_image(self, image):
        """Store camera image in buffer for main thread to render."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha
        array = array[:, :, ::-1]  # BGR to RGB
        self.image_data = array
        
    def _render_frame(self):
        """Render the current frame (call from main thread only)."""
        if self.display is None or self.image_data is None:
            return
        
        try:
            surface = pygame.surfarray.make_surface(self.image_data.swapaxes(0, 1))
            self.display.blit(surface, (0, 0))
            
            # Draw HUD
            self._draw_hud()
            
            pygame.display.flip()
        except Exception as e:
            pass  # Ignore display errors
        
    def _draw_hud(self):
        """Draw heads-up display."""
        if self.ego_vehicle is None:
            return
            
        font = pygame.font.Font(None, 36)
        
        speed = get_speed_kmh(self.ego_vehicle)
        current_loc = self.ego_vehicle.get_location()
        distance = get_distance_traveled(self.start_location, current_loc)
        
        # Speed display
        speed_text = font.render(f"Speed: {speed:.1f} km/h", True, (255, 255, 255))
        self.display.blit(speed_text, (10, 10))
        
        # Distance display
        dist_text = font.render(f"Distance: {distance:.1f} m", True, (255, 255, 255))
        self.display.blit(dist_text, (10, 45))
        
        # Zone display
        zone = "Zone 1 (Accel)" if not self.zone1_complete else "Zone 2/3 (Brake)"
        zone_text = font.render(zone, True, (255, 255, 0))
        self.display.blit(zone_text, (10, 80))
        
        # Obstacle info
        if self.final_distance is not None:
            result_text = font.render(f"Stopped at: {self.final_distance:.1f} m", True, (0, 255, 0))
            self.display.blit(result_text, (10, 115))
            
    def run_test(self, wait_for_obstacle=True, wait_time=5.0):
        """Run single AEB test."""
        if wait_for_obstacle:
            print(f"Waiting {wait_time}s for obstacle spawner...")
            start_wait = time.time()
            while time.time() - start_wait < wait_time:
                self.world.tick()
                self._render_frame()  # Render during wait
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None
        
        print("\n" + "="*50)
        print("STARTING AEB TEST")
        print("="*50)
        print(f"Zone 1: Accelerate to {CONFIG['target_speed_kmh']} km/h over {CONFIG['zone1_distance']}m")
        print(f"Target stop distance: {CONFIG['target_stop_distance']}m (+/- 3m)")
        print("="*50 + "\n")
        
        # Control loop
        while not self.test_complete:
            self.world.tick()
            
            # Render camera frame from main thread
            self._render_frame()
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
            
            current_loc = self.ego_vehicle.get_location()
            distance_traveled = get_distance_traveled(self.start_location, current_loc)
            current_speed = get_speed_kmh(self.ego_vehicle)
            
            # Find obstacle ahead
            obstacle, obstacle_distance = find_obstacle_ahead(self.ego_vehicle, self.world)
            
            # Determine control action
            control = carla.VehicleControl()
            
            # ZONE 1: Acceleration Phase
            if distance_traveled < CONFIG['zone1_distance'] and obstacle is None:
                if current_speed < CONFIG['target_speed_kmh']:
                    # Gentle acceleration
                    speed_diff = CONFIG['target_speed_kmh'] - current_speed
                    control.throttle = min(0.7, 0.3 + speed_diff / 30.0)
                else:
                    control.throttle = 0.3  # Maintain speed
                control.brake = 0.0
                
            # ZONE 2/3: Braking Phase
            elif obstacle is not None:
                if not self.obstacle_detected:
                    print(f"OBSTACLE DETECTED at {obstacle_distance:.1f}m")
                    print(f"Current speed: {current_speed:.1f} km/h")
                    self.obstacle_detected = True
                    self.zone1_complete = True
                
                # Calculate brake force (returns 0 if obstacle is far)
                brake_force = calculate_brake_force(obstacle_distance, current_speed)
                
                # If obstacle is far, continue at target speed
                if brake_force == 0.0:
                    if current_speed < CONFIG['target_speed_kmh']:
                        control.throttle = 0.5
                    else:
                        control.throttle = 0.3
                    control.brake = 0.0
                else:
                    # Apply brakes when obstacle is close
                    control.throttle = 0.0
                    control.brake = brake_force
                
                # Check if stopped (only after car has moved at least 5m)
                if current_speed < 0.5 and distance_traveled > 5.0:
                    self.final_distance = obstacle_distance
                    self.test_complete = True
                    
                    # Result evaluation
                    print("\n" + "="*50)
                    print("TEST COMPLETE")
                    print("="*50)
                    print(f"Final distance to obstacle: {self.final_distance:.1f}m")
                    print(f"Target: {CONFIG['target_stop_distance']}m (+/- 3m)")
                    
                    if 3.0 <= self.final_distance <= 9.0:
                        print("RESULT: PASS ✓")
                    else:
                        print("RESULT: FAIL ✗")
                    print("="*50)
                    
            else:
                # Continue accelerating if in zone 1 but no obstacle yet
                if current_speed < CONFIG['target_speed_kmh']:
                    control.throttle = 0.5
                else:
                    control.throttle = 0.3
                control.brake = 0.0
                self.zone1_complete = True
            
            self.ego_vehicle.apply_control(control)
            
        return self.final_distance
        
    def cleanup(self):
        """Clean up actors and reset world."""
        if self.camera:
            self.camera.destroy()
        if self.ego_vehicle:
            self.ego_vehicle.destroy()
        
        # Reset synchronous mode
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        
        pygame.quit()


def main():
    """Main entry point."""
    controller = AEBController()
    
    try:
        controller.connect()
        
        if not controller.spawn_ego_vehicle():
            return
        
        controller.setup_camera()
        
        # Run test
        result = controller.run_test(wait_for_obstacle=True, wait_time=3.0)
        
        if result is not None:
            print(f"\nFinal stopping distance: {result:.1f}m")
        
        # Keep display open briefly
        time.sleep(2.0)
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        controller.cleanup()


if __name__ == '__main__':
    main()
