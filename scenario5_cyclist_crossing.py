#!/usr/bin/env python3
"""
AEB Scenario 5: Car-to-Cyclist Nearside Obstructed (CCNO)
==========================================================
BAJA SAEINDIA 2025 - Autonomous Emergency Braking

Scenario: Cyclist suddenly crosses from behind obstruction.
- Cyclist waits hidden at random 110-150m ahead (on roadside)
- Ego accelerates to 30 km/h (Zone 1: 100m)
- When ego is within 40m, cyclist starts crossing
- Cyclist moves faster than pedestrian (~15 km/h)
- Ego must detect and brake to avoid collision
- Target: Stop within 6m ± 2m edge-to-edge

Usage: python3 scenario5_cyclist_crossing.py
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
    'cyclist_min_distance': 110.0,
    'cyclist_max_distance': 150.0,
    'crossing_trigger_distance': 40.0,   # Cyclist crosses when ego is this close
    'cyclist_speed': 4.2,                 # m/s (~15 km/h cycling)
    'target_stop_distance': 6.0,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_speed_kmh(vehicle) -> float:
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6


def get_distance(loc1, loc2) -> float:
    return math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)


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

class Scenario5Runner:
    def __init__(self):
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.cyclist = None
        self.camera = None
        self.display = None
        self.aeb = None
        self.image_data = None
        self.start_location = None
        self.cyclist_location = None
        self.cyclist_crossing = False
        self.cyclist_target = None
        self.cyclist_waypoint = None
        self.crossing_point = None  # Point on road where cyclist will cross
        
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
        print("SCENARIO 5: Cyclist Nearside Obstructed (CCNO)")
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
        
        # Spawn cyclist at RANDOM distance, offset to the side
        cyclist_distance = random.uniform(CONFIG['cyclist_min_distance'], CONFIG['cyclist_max_distance'])
        carla_map = self.world.get_map()
        ego_wp = carla_map.get_waypoint(spawn_point.location)
        cyclist_wps = ego_wp.next(cyclist_distance)
        
        if cyclist_wps:
            cyclist_wp = cyclist_wps[0]
            self.cyclist_waypoint = cyclist_wp
            
            # Get right vector for offset (perpendicular to road)
            right_vec = cyclist_wp.transform.get_right_vector()
            
            # Start position: 5m to the right of lane center (hidden behind imaginary obstruction)
            start_offset = 5.0
            
            # Calculate yaw to face LEFT (towards the road center, crossing direction)
            # Adding 90 to road yaw points perpendicular, but we need LEFT not RIGHT
            # So we use -90 to face towards the road
            crossing_yaw = cyclist_wp.transform.rotation.yaw - 90
            
            cyclist_spawn = carla.Transform(
                carla.Location(
                    x=cyclist_wp.transform.location.x + right_vec.x * start_offset,
                    y=cyclist_wp.transform.location.y + right_vec.y * start_offset,
                    z=cyclist_wp.transform.location.z + 0.3
                ),
                carla.Rotation(yaw=crossing_yaw)
            )
            
            # Store crossing point (road center) for trigger distance calculation
            self.crossing_point = carla.Location(
                x=cyclist_wp.transform.location.x,
                y=cyclist_wp.transform.location.y,
                z=cyclist_wp.transform.location.z
            )
            
            # Target: 5m to the LEFT of road center (cross fully across)
            self.cyclist_target = carla.Location(
                x=cyclist_wp.transform.location.x - right_vec.x * 5.0,
                y=cyclist_wp.transform.location.y - right_vec.y * 5.0,
                z=cyclist_wp.transform.location.z
            )
            
            # Spawn cyclist (bike)
            cyclist_bps = bp_lib.filter('vehicle.bh.crossbike')
            if not cyclist_bps:
                cyclist_bps = bp_lib.filter('vehicle.diamondback.century')
            if not cyclist_bps:
                cyclist_bps = bp_lib.filter('vehicle.gazelle.omafiets')
            if not cyclist_bps:
                cyclist_bps = bp_lib.filter('*bike*')
            
            if cyclist_bps:
                cyclist_bp = random.choice(cyclist_bps)
                self.cyclist = self.world.spawn_actor(cyclist_bp, cyclist_spawn)
                self.cyclist_location = cyclist_spawn.location
                print(f"Cyclist spawned at unknown distance (hidden, will cross)")
            else:
                print("WARNING: No bike blueprint found!")
                return False
        
        # AEB config
        aeb_config = AEBConfig(
            zone1_distance=100.0,
            target_speed_kmh=30.0,
            brake_start_distance=14.0,  # Slightly larger for faster crossing
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
        pygame.display.set_caption('AEB - Scenario 5: Cyclist Crossing')
        
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
                f"Cyclist: {'CROSSING!' if self.cyclist_crossing else 'Hidden'}",
                f"Distance: {state['obstacle_distance']:.1f}m" if state['obstacle_distance'] else "Distance: --",
            ]
            
            for i, text in enumerate(texts):
                color = (255, 165, 0) if self.cyclist_crossing else (255, 255, 255)  # Orange for cyclist
                rendered = font.render(text, True, color)
                self.display.blit(rendered, (10, 10 + i * 35))
            
            pygame.display.flip()
        except:
            pass
            
    def run(self):
        print("\nStarting test - Cyclist will cross suddenly...")
        
        test_complete = False
        final_distance = None
        collision = False
        
        while not test_complete:
            self.world.tick()
            self._render()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
            
            # Check if cyclist should start crossing
            if not self.cyclist_crossing and self.cyclist and self.crossing_point:
                ego_loc = self.ego_vehicle.get_location()
                
                # Distance to crossing point (road center)
                dist_to_crossing = get_distance(ego_loc, self.crossing_point)
                
                if dist_to_crossing < CONFIG['crossing_trigger_distance']:
                    self.cyclist_crossing = True
                    print(f">>> CYCLIST CROSSING! (ego at {dist_to_crossing:.1f}m from crossing)")
            
            # Keep cyclist moving if crossing - just go straight (cyclist is already oriented to cross)
            if self.cyclist_crossing and self.cyclist:
                # Cyclist just goes forward (perpendicular to road, towards center then past)
                self.cyclist.apply_control(carla.VehicleControl(throttle=0.7, steer=0.0, brake=0.0))
            
            # AEB control
            control = self.aeb.update(self.ego_vehicle, self.world)
            self.ego_vehicle.apply_control(control)
            
            speed = get_speed_kmh(self.ego_vehicle)
            state = self.aeb.get_state()
            
            # Check collision
            if state['obstacle_distance'] and state['obstacle_distance'] < 0.5:
                collision = True
                final_distance = state['obstacle_distance']
                test_complete = True
            # Check stopped
            elif state['stopped'] or (speed < 0.5 and self.cyclist_crossing and state['state'] == 'ZONE3_BRAKING'):
                if state['obstacle_distance']:
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
        if self.cyclist:
            self.cyclist.destroy()
        
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        
        pygame.quit()


def main():
    runner = Scenario5Runner()
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
