#!/usr/bin/env python3
"""
AEB Scenario 2: Car-to-Car Rear Braking (CCRb)
===============================================
BAJA SAEINDIA 2025 - Autonomous Emergency Braking

Scenario: Ego follows lead vehicle that suddenly brakes.
- Lead vehicle spawned at random 110-150m ahead, moving at 30 km/h
- Ego accelerates to 30 km/h (Zone 1: 100m)
- After 100m, ego cruises and follows lead
- Lead suddenly brakes hard
- Ego must detect closing gap and brake to avoid collision
- Target: Stop within 6m ± 2m edge-to-edge

Usage: python3 scenario2_car_braking.py
"""

import carla
import pygame
import numpy as np
import math
import random
import time

from aeb_core import AEBController, AEBConfig

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    'host': 'localhost',
    'port': 2000,
    'timeout': 10.0,
    'lead_min_distance': 110.0,    # Random spawn range start
    'lead_max_distance': 150.0,    # Random spawn range end
    'lead_speed_kmh': 30.0,        # Lead vehicle cruising speed
    'brake_trigger_distance': 100.0,  # Ego traveled distance when lead brakes
    'target_stop_distance': 6.0,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_speed_kmh(vehicle) -> float:
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6


def find_straight_spawn(world, min_length: float = 200.0):
    """Find straight spawn using known good indices for Town04_Opt."""
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    
    # Known straight spawns in Town04_Opt
    preferred_indices = [119, 121, 155, 157]
    
    for idx in preferred_indices:
        if idx < len(spawn_points):
            sp = spawn_points[idx]
            wp = carla_map.get_waypoint(sp.location)
            if wp and not wp.is_junction:
                print(f"Using straight spawn #{idx}")
                return sp
    
    # Fallback
    for sp in spawn_points:
        wp = carla_map.get_waypoint(sp.location)
        if wp and not wp.is_junction and wp.lane_width > 3.0:
            return sp
    
    return spawn_points[0]


def clear_area(world, location, radius=150.0):
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.get_location().distance(location) < radius:
            actor.destroy()
    for actor in world.get_actors().filter('walker.*'):
        if actor.get_location().distance(location) < radius:
            actor.destroy()


# =============================================================================
# SCENARIO RUNNER
# =============================================================================

class Scenario2Runner:
    def __init__(self):
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.lead_vehicle = None
        self.camera = None
        self.display = None
        self.aeb = None
        self.image_data = None
        self.start_location = None
        self.lead_braking = False
        
    def connect(self):
        print("Connecting to CARLA...")
        self.client = carla.Client(CONFIG['host'], CONFIG['port'])
        self.client.set_timeout(CONFIG['timeout'])
        self.world = self.client.get_world()
        
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        
        print(f"Connected - Map: {self.world.get_map().name}")
        
    def setup(self):
        print("\n" + "="*60)
        print("SCENARIO 2: Car-to-Car Rear Braking (CCRb)")
        print("="*60)
        
        spawn_point = find_straight_spawn(self.world)
        if not spawn_point:
            return False
        
        clear_area(self.world, spawn_point.location)
        self.world.tick()
        
        bp_lib = self.world.get_blueprint_library()
        
        # Spawn ego
        ego_bp = bp_lib.find('vehicle.tesla.model3')
        self.ego_vehicle = self.world.spawn_actor(ego_bp, spawn_point)
        self.start_location = spawn_point.location
        print(f"Ego spawned at ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
        
        # Spawn lead vehicle at RANDOM distance
        lead_distance = random.uniform(CONFIG['lead_min_distance'], CONFIG['lead_max_distance'])
        carla_map = self.world.get_map()
        ego_wp = carla_map.get_waypoint(spawn_point.location)
        lead_wps = ego_wp.next(lead_distance)
        
        if lead_wps:
            lead_transform = lead_wps[0].transform
            lead_transform.location.z += 0.5
            
            lead_bp = bp_lib.find('vehicle.audi.a2')
            self.lead_vehicle = self.world.spawn_actor(lead_bp, lead_transform)
            print(f"Lead vehicle spawned at unknown distance (ego must detect)")
        
        # AEB config
        aeb_config = AEBConfig(
            zone1_distance=100.0,
            target_speed_kmh=30.0,
            brake_start_distance=12.0,
            target_stop_distance=6.0,
            emergency_distance=4.0,
            min_brake=0.4,
            max_brake=1.0,
        )
        self.aeb = AEBController(aeb_config)
        
        self._setup_camera()
        
        # Prime lead vehicle to start moving IMMEDIATELY
        # Apply several ticks with throttle so lead is moving when ego starts
        for _ in range(40):  # 2 seconds of simulation time
            self.lead_vehicle.apply_control(carla.VehicleControl(throttle=0.8, brake=0.0))
            self.world.tick()
        
        print(f"Lead vehicle primed - speed: {get_speed_kmh(self.lead_vehicle):.1f} km/h")
        
        return True
        
    def _setup_camera(self):
        pygame.init()
        self.display = pygame.display.set_mode((800, 600))
        pygame.display.set_caption('AEB - Scenario 2: CCRb')
        
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
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
        self.image_data = array
        
    def _render(self):
        if self.display is None or self.image_data is None:
            return
        try:
            surface = pygame.surfarray.make_surface(self.image_data.swapaxes(0, 1))
            self.display.blit(surface, (0, 0))
            
            font = pygame.font.Font(None, 36)
            ego_speed = get_speed_kmh(self.ego_vehicle)
            lead_speed = get_speed_kmh(self.lead_vehicle) if self.lead_vehicle else 0
            state = self.aeb.get_state()
            
            texts = [
                f"Ego Speed: {ego_speed:.1f} km/h",
                f"Lead Speed: {lead_speed:.1f} km/h",
                f"State: {state['state']}",
                f"Lead: {'BRAKING!' if self.lead_braking else 'Moving'}",
                f"Gap: {state['obstacle_distance']:.1f}m" if state['obstacle_distance'] else "Gap: --",
            ]
            
            for i, text in enumerate(texts):
                color = (255, 0, 0) if self.lead_braking else (255, 255, 255)
                rendered = font.render(text, True, color)
                self.display.blit(rendered, (10, 10 + i * 35))
            
            pygame.display.flip()
        except:
            pass
            
    def run(self):
        print("\nStarting test - Lead will brake suddenly...")
        
        test_complete = False
        final_distance = None
        collision = False
        
        while not test_complete:
            self.world.tick()
            self._render()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
            
            state = self.aeb.get_state()
            
            # Control lead vehicle
            if not self.lead_braking:
                # Lead accelerates to cruising speed
                lead_speed = get_speed_kmh(self.lead_vehicle)
                if lead_speed < CONFIG['lead_speed_kmh']:
                    self.lead_vehicle.apply_control(carla.VehicleControl(throttle=0.6))
                else:
                    self.lead_vehicle.apply_control(carla.VehicleControl(throttle=0.3))
                
                # Trigger braking after ego travels 100m
                if state['distance_traveled'] > CONFIG['brake_trigger_distance']:
                    self.lead_braking = True
                    print(f">>> LEAD BRAKING! (ego traveled {state['distance_traveled']:.1f}m)")
            else:
                # Lead braking hard
                self.lead_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            
            # AEB control
            control = self.aeb.update(self.ego_vehicle, self.world)
            self.ego_vehicle.apply_control(control)
            
            # Check completion
            ego_speed = get_speed_kmh(self.ego_vehicle)
            
            if state['obstacle_distance']:
                if state['obstacle_distance'] < 1.0:
                    collision = True
                    final_distance = state['obstacle_distance']
                    test_complete = True
                elif state['stopped'] or (ego_speed < 0.5 and self.lead_braking):
                    final_distance = state['obstacle_distance']
                    test_complete = True
        
        # Results
        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)
        print(f"Edge-to-edge distance: {final_distance:.1f}m")
        print(f"Target: 6.0m (+/- 2m)")
        
        if collision:
            print("RESULT: FAIL ✗ (COLLISION!)")
        elif 4.0 <= final_distance <= 8.0:
            print("RESULT: PASS ✓")
        else:
            print("RESULT: FAIL ✗")
        print("="*60)
        
        time.sleep(2)
        return final_distance
        
    def cleanup(self):
        if self.camera:
            self.camera.destroy()
        if self.ego_vehicle:
            self.ego_vehicle.destroy()
        if self.lead_vehicle:
            self.lead_vehicle.destroy()
        
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        
        pygame.quit()


def main():
    runner = Scenario2Runner()
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
