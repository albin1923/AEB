#!/usr/bin/env python3
"""
AEB Scenario 1: Car-to-Car Rear Stationary
===========================================
BAJA SAEINDIA 2025 - Autonomous Emergency Braking

Scenario: Ego vehicle approaches a stationary vehicle ahead.
- Ego accelerates to 30 km/h
- Detects stationary car ahead
- Brakes gradually to stop within 6m (+/- 3m)

Usage: python3 scenario1_car_stationary.py
"""

import carla
import pygame
import numpy as np
import math
import random
import time
import sys

from aeb_core import AEBController, AEBConfig

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    'host': 'localhost',
    'port': 2000,
    'timeout': 10.0,
    'obstacle_min_distance': 110.0,  # Minimum: 100m accel + 10m
    'obstacle_max_distance': 150.0,  # Maximum: 100m accel + 50m
    'target_stop_distance': 6.0,     # target: 6m ± 2m (4-8m acceptable)
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_speed_kmh(vehicle) -> float:
    """Get vehicle speed in km/h."""
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6


def find_straight_spawn(world, min_length: float = 120.0):
    """
    Find a spawn point on a perfectly straight highway segment.
    Uses known good spawn indices for Town04_Opt.
    """
    carla_map = world.get_map()
    spawn_points = world.get_map().get_spawn_points()
    
    # Known straight spawns in Town04_Opt (verified: 0.16° deviation over 120m)
    # #119: (65.3, 16.8), #121: (65.3, 9.8), #155: (-67.1, 33.9), #157: (-67.1, 26.9)
    preferred_indices = [119, 121, 155, 157]
    
    for idx in preferred_indices:
        if idx < len(spawn_points):
            sp = spawn_points[idx]
            wp = carla_map.get_waypoint(sp.location)
            if wp and not wp.is_junction:
                print(f"Using known straight spawn #{idx} at ({sp.location.x:.1f}, {sp.location.y:.1f})")
                return sp
    
    # Fallback: search for straight roads
    best_spawn = None
    best_dev = 999
    
    for sp in spawn_points:
        waypoint = carla_map.get_waypoint(sp.location)
        if waypoint is None or waypoint.is_junction or waypoint.lane_width < 3.2:
            continue
        
        initial_yaw = waypoint.transform.rotation.yaw
        check_wp = waypoint
        max_dev = 0
        dist = 0
        valid = True
        
        for _ in range(60):
            next_wps = check_wp.next(2.0)
            if not next_wps:
                valid = False
                break
            check_wp = next_wps[0]
            dist += 2
            if check_wp.is_junction:
                valid = False
                break
            yaw_diff = abs(check_wp.transform.rotation.yaw - initial_yaw)
            if yaw_diff > 180:
                yaw_diff = 360 - yaw_diff
            max_dev = max(max_dev, yaw_diff)
            if yaw_diff > 2.0:
                valid = False
                break
        
        if valid and dist >= min_length and max_dev < best_dev:
            best_dev = max_dev
            best_spawn = sp
    
    if best_spawn:
        print(f"Found straight road (deviation: {best_dev:.2f}°)")
        return best_spawn
    
    print("WARNING: No straight road found, using first spawn")
    return spawn_points[0] if spawn_points else None


def clear_area(world, location, radius: float = 100.0):
    """Remove vehicles and walkers near location."""
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.get_location().distance(location) < radius:
            actor.destroy()
    for actor in world.get_actors().filter('walker.*'):
        if actor.get_location().distance(location) < radius:
            actor.destroy()


# =============================================================================
# SCENARIO RUNNER
# =============================================================================

class Scenario1Runner:
    def __init__(self):
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.obstacle = None
        self.camera = None
        self.display = None
        self.aeb = None
        self.image_data = None
        self.start_location = None
        
    def connect(self):
        """Connect to CARLA."""
        print("Connecting to CARLA...")
        self.client = carla.Client(CONFIG['host'], CONFIG['port'])
        self.client.set_timeout(CONFIG['timeout'])
        self.world = self.client.get_world()
        
        # Enable sync mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        
        print(f"Connected - Map: {self.world.get_map().name}")
        
    def setup(self):
        """Setup scenario."""
        print("\n" + "="*60)
        print("SCENARIO 1: Car-to-Car Rear Stationary")
        print("="*60)
        
        # Find straight road
        spawn_point = find_straight_spawn(self.world)
        if not spawn_point:
            return False
        
        # Clear area
        clear_area(self.world, spawn_point.location)
        self.world.tick()
        
        # Spawn ego vehicle
        bp_lib = self.world.get_blueprint_library()
        ego_bp = bp_lib.find('vehicle.tesla.model3')
        self.ego_vehicle = self.world.spawn_actor(ego_bp, spawn_point)
        self.start_location = spawn_point.location
        print(f"Ego spawned at ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
        
        # Spawn stationary obstacle at RANDOM distance (ego doesn't know where)
        obstacle_distance = random.uniform(CONFIG['obstacle_min_distance'], CONFIG['obstacle_max_distance'])
        carla_map = self.world.get_map()
        ego_wp = carla_map.get_waypoint(spawn_point.location)
        obstacle_wps = ego_wp.next(obstacle_distance)
        
        if obstacle_wps:
            obstacle_transform = obstacle_wps[0].transform
            obstacle_transform.location.z += 0.5
            
            obstacle_bp = bp_lib.find('vehicle.audi.a2')
            self.obstacle = self.world.spawn_actor(obstacle_bp, obstacle_transform)
            self.obstacle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
            print(f"Obstacle spawned at unknown distance (ego must detect)")
        
        # Initialize AEB controller with 3-zone config
        aeb_config = AEBConfig(
            zone1_distance=100.0,         # Accelerate for 100m
            target_speed_kmh=30.0,
            brake_start_distance=12.0,    # Start braking at 12m edge-to-edge
            target_stop_distance=6.0,     # Target stop at 6m
            emergency_distance=4.0,
            min_brake=0.4,
            max_brake=1.0,
        )
        self.aeb = AEBController(aeb_config)
        
        # Setup camera
        self._setup_camera()
        
        self.world.tick()
        return True
        
    def _setup_camera(self):
        """Setup pygame display and camera."""
        pygame.init()
        self.display = pygame.display.set_mode((800, 600))
        pygame.display.set_caption('AEB - Scenario 1: Car-to-Car Rear Stationary')
        
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(
            carla.Location(x=-8, z=4),
            carla.Rotation(pitch=-15)
        )
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
        self.camera.listen(lambda img: self._store_image(img))
        
    def _store_image(self, image):
        """Store camera image."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
        self.image_data = array
        
    def _render(self):
        """Render frame with HUD."""
        if self.display is None or self.image_data is None:
            return
        try:
            surface = pygame.surfarray.make_surface(self.image_data.swapaxes(0, 1))
            self.display.blit(surface, (0, 0))
            
            # HUD
            font = pygame.font.Font(None, 36)
            speed = get_speed_kmh(self.ego_vehicle)
            state = self.aeb.get_state()
            
            texts = [
                f"Speed: {speed:.1f} km/h",
                f"State: {state['state']}",
                f"Obstacle: {state['obstacle_distance']:.1f}m" if state['obstacle_distance'] else "Obstacle: None",
                f"TTC: {state['ttc']:.1f}s" if state['ttc'] else "TTC: --",
            ]
            
            for i, text in enumerate(texts):
                color = (255, 255, 0) if 'BRAKING' in state['state'] else (255, 255, 255)
                rendered = font.render(text, True, color)
                self.display.blit(rendered, (10, 10 + i * 35))
            
            pygame.display.flip()
        except Exception:
            pass
            
    def run(self):
        """Run scenario."""
        print("\nStarting test...")
        
        test_complete = False
        final_distance = None
        
        while not test_complete:
            self.world.tick()
            self._render()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
            
            # Get AEB control
            control = self.aeb.update(self.ego_vehicle, self.world)
            self.ego_vehicle.apply_control(control)
            
            # Check if stopped
            speed = get_speed_kmh(self.ego_vehicle)
            state = self.aeb.get_state()
            
            # Check stopped state from AEB or low speed near obstacle
            if state['stopped'] or (speed < 0.5 and state['state'] == 'BRAKING'):
                if state['obstacle_distance']:
                    final_distance = state['obstacle_distance']
                    test_complete = True
        
        # Results (distance is edge-to-edge)
        
        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)
        print(f"Edge-to-edge distance: {final_distance:.1f}m")
        print(f"Target: {CONFIG['target_stop_distance']}m (+/- 2m)")
        
        if 4.0 <= final_distance <= 8.0:
            print("RESULT: PASS ✓")
        else:
            print("RESULT: FAIL ✗")
        print("="*60)
        
        time.sleep(2)
        return final_distance
        
    def cleanup(self):
        """Cleanup actors."""
        if self.camera:
            self.camera.destroy()
        if self.ego_vehicle:
            self.ego_vehicle.destroy()
        if self.obstacle:
            self.obstacle.destroy()
        
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        
        pygame.quit()


def main():
    runner = Scenario1Runner()
    try:
        runner.connect()
        if runner.setup():
            runner.run()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        runner.cleanup()


if __name__ == '__main__':
    main()
