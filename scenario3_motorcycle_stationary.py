#!/usr/bin/env python3
"""
AEB Scenario 3: Car-to-Motorcycle Stationary
=============================================
BAJA SAEINDIA 2025 - Autonomous Emergency Braking

Scenario: Ego approaches stationary motorcycle on road.
- Motorcycle spawned at random 110-150m ahead
- Ego accelerates to 30 km/h (Zone 1: 100m)
- After 100m, ego cruises and detects motorcycle
- Ego must brake to avoid collision
- Target: Stop within 6m ± 2m edge-to-edge

Usage: python3 scenario3_motorcycle_stationary.py
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
    'obstacle_min_distance': 110.0,
    'obstacle_max_distance': 150.0,
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
    
    preferred_indices = [119, 121, 155, 157]
    
    for idx in preferred_indices:
        if idx < len(spawn_points):
            sp = spawn_points[idx]
            wp = carla_map.get_waypoint(sp.location)
            if wp and not wp.is_junction:
                print(f"Using straight spawn #{idx}")
                return sp
    
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

class Scenario3Runner:
    def __init__(self):
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.motorcycle = None
        self.camera = None
        self.display = None
        self.aeb = None
        self.image_data = None
        self.start_location = None
        
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
        print("SCENARIO 3: Car-to-Motorcycle Stationary")
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
        
        # Spawn motorcycle at RANDOM distance
        obstacle_distance = random.uniform(CONFIG['obstacle_min_distance'], CONFIG['obstacle_max_distance'])
        carla_map = self.world.get_map()
        ego_wp = carla_map.get_waypoint(spawn_point.location)
        obstacle_wps = ego_wp.next(obstacle_distance)
        
        if obstacle_wps:
            obstacle_transform = obstacle_wps[0].transform
            obstacle_transform.location.z += 0.3
            
            # Find motorcycle blueprint
            moto_bps = [bp for bp in bp_lib.filter('vehicle.*') if 'motorcycle' in bp.id.lower() or 'harley' in bp.id.lower() or 'kawasaki' in bp.id.lower() or 'yamaha' in bp.id.lower()]
            if moto_bps:
                moto_bp = random.choice(moto_bps)
            else:
                moto_bp = bp_lib.find('vehicle.harley-davidson.low_rider')
            
            self.motorcycle = self.world.spawn_actor(moto_bp, obstacle_transform)
            self.motorcycle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
            print(f"Motorcycle spawned at unknown distance (ego must detect)")
        
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
        self.world.tick()
        return True
        
    def _setup_camera(self):
        pygame.init()
        self.display = pygame.display.set_mode((800, 600))
        pygame.display.set_caption('AEB - Scenario 3: Motorcycle Stationary')
        
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
            speed = get_speed_kmh(self.ego_vehicle)
            state = self.aeb.get_state()
            
            texts = [
                f"Speed: {speed:.1f} km/h",
                f"State: {state['state']}",
                f"Traveled: {state['distance_traveled']:.1f}m",
                f"Motorcycle: {state['obstacle_distance']:.1f}m" if state['obstacle_distance'] else "Motorcycle: Not detected",
            ]
            
            for i, text in enumerate(texts):
                color = (255, 255, 0) if 'BRAKING' in state['state'] else (255, 255, 255)
                rendered = font.render(text, True, color)
                self.display.blit(rendered, (10, 10 + i * 35))
            
            pygame.display.flip()
        except:
            pass
            
    def run(self):
        print("\nStarting test...")
        
        test_complete = False
        final_distance = None
        
        while not test_complete:
            self.world.tick()
            self._render()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
            
            control = self.aeb.update(self.ego_vehicle, self.world)
            self.ego_vehicle.apply_control(control)
            
            speed = get_speed_kmh(self.ego_vehicle)
            state = self.aeb.get_state()
            
            if state['stopped'] or (speed < 0.5 and state['state'] == 'ZONE3_BRAKING'):
                if state['obstacle_distance']:
                    final_distance = state['obstacle_distance']
                    test_complete = True
        
        # Results
        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)
        print(f"Edge-to-edge distance: {final_distance:.1f}m")
        print(f"Target: 6.0m (+/- 2m)")
        
        if 4.0 <= final_distance <= 8.0:
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
        if self.motorcycle:
            self.motorcycle.destroy()
        
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        
        pygame.quit()


def main():
    runner = Scenario3Runner()
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
