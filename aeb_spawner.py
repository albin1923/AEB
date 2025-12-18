#!/usr/bin/env python3
"""
AEB Obstacle Spawner - Robust Two-File Architecture
====================================================
This script spawns a stationary obstacle for the AEB controller.

USAGE:
------
Terminal 1: python3 aeb_main.py
Terminal 2: python3 aeb_spawner.py (run within 15 seconds)

This script:
- Reads spawn info from the main controller
- Spawns ONE stationary vehicle 50-100m ahead
- Uses the SAME lane as the ego vehicle
- Applies handbrake and exits (vehicle persists)
"""

import carla
import math
import time
import random
import json
import os

# =============================================================================
# CONFIGURATION - Must match aeb_main.py
# =============================================================================
CONFIG = {
    'host': '127.0.0.1',
    'port': 2000,
    'timeout': 10.0,
    'spawn_info_file': '/tmp/aeb_spawn_info.json',
    
    # Obstacle spawn range (meters ahead of ego spawn)
    'min_distance': 60,
    'max_distance': 100,
}


def read_spawn_info(timeout=10):
    """Read spawn info from main controller"""
    start = time.time()
    
    while time.time() - start < timeout:
        if os.path.exists(CONFIG['spawn_info_file']):
            try:
                with open(CONFIG['spawn_info_file'], 'r') as f:
                    data = json.load(f)
                    if data.get('ready', False):
                        return data
            except:
                pass
        time.sleep(0.5)
    
    return None


def main():
    print("=" * 60)
    print("AEB OBSTACLE SPAWNER")
    print("=" * 60)
    
    # Read spawn info from main controller
    print("[WAIT] Waiting for main controller...")
    spawn_info = read_spawn_info()
    
    if spawn_info is None:
        print("[ERROR] Could not read spawn info!")
        print("[ERROR] Make sure aeb_main.py is running first.")
        return
    
    print(f"[INFO] Received spawn info from main controller")
    
    # Connect to CARLA
    client = carla.Client(CONFIG['host'], CONFIG['port'])
    client.set_timeout(CONFIG['timeout'])
    world = client.get_world()
    
    # DO NOT change world settings - main controller handles sync mode
    print("[INFO] Connected to CARLA (using existing settings)")
    
    blueprint_library = world.get_blueprint_library()
    map_obj = world.get_map()
    
    # Get ego spawn location
    ego_location = carla.Location(
        x=spawn_info['x'],
        y=spawn_info['y'],
        z=spawn_info['z']
    )
    
    # Get waypoint at ego location
    ego_waypoint = map_obj.get_waypoint(ego_location)
    
    # Calculate obstacle spawn distance
    spawn_attempts = 5
    obstacle = None
    
    for attempt in range(spawn_attempts):
        obstacle_distance = random.randint(CONFIG['min_distance'], CONFIG['max_distance'])
        print(f"[INFO] Attempt {attempt+1}: Spawning obstacle {obstacle_distance}m ahead...")
        
        # Get waypoint ahead
        waypoints_ahead = ego_waypoint.next(float(obstacle_distance))
        
        if not waypoints_ahead:
            print("[WARN] Could not find waypoint ahead, trying different distance...")
            continue
        
        obstacle_waypoint = waypoints_ahead[0]
        obstacle_transform = obstacle_waypoint.transform
        obstacle_transform.location.z += 0.5  # Slightly higher lift
        
        # Select random vehicle type
        vehicle_types = [
            'vehicle.audi.a2',
            'vehicle.bmw.grandtourer',
            'vehicle.citroen.c3',
            'vehicle.nissan.micra',
            'vehicle.seat.leon'
        ]
        vehicle_type = random.choice(vehicle_types)
        vehicle_bp = blueprint_library.find(vehicle_type)
        
        # Try to spawn
        obstacle = world.try_spawn_actor(vehicle_bp, obstacle_transform)
        
        if obstacle is not None:
            break
        else:
            print(f"[WARN] Location {obstacle_distance}m ahead is occupied, retrying...")
    
    if obstacle is None:
        print("[ERROR] Failed to spawn obstacle after all attempts!")
        return
    
    # Apply stationary controls
    obstacle.apply_control(carla.VehicleControl(
        throttle=0.0,
        brake=1.0,
        hand_brake=True
    ))
    
    # Set velocity to zero
    obstacle.set_target_velocity(carla.Vector3D(0, 0, 0))
    
    # Calculate actual distance from ego spawn
    actual_dist = math.sqrt(
        (obstacle_transform.location.x - ego_location.x)**2 +
        (obstacle_transform.location.y - ego_location.y)**2
    )
    
    print("\n" + "=" * 60)
    print(f"SUCCESS: {vehicle_type.split('.')[-1].upper()} spawned!")
    print(f"Distance from ego spawn: {actual_dist:.1f}m")
    print(f"Location: ({obstacle_transform.location.x:.1f}, {obstacle_transform.location.y:.1f})")
    print("=" * 60)
    print("\nObstacle will remain in world.")
    print("Main controller will detect and brake.\n")


if __name__ == '__main__':
    main()
