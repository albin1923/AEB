#!/usr/bin/env python3
"""
AEB System with 3 Zones:
- Zone 1 (0-30m): Pure acceleration - build up speed
- Zone 2 (30m+): RADAR scanning, gradual deceleration when obstacle detected
- Zone 3 (within 6m): Complete stop

The car does NOT know obstacle position initially - only RADAR detection triggers braking
"""

import carla
import numpy as np
import time
import pygame
import random
import math

def main():
    print("=" * 60)
    print("AEB SYSTEM - 3 ZONE OPERATION")
    print("Zone 1: Acceleration (0-30m)")
    print("Zone 2: RADAR Scanning & Gradual Braking")
    print("Zone 3: Stop within 6m of obstacle")
    print("=" * 60)
    
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    settings = world.get_settings()
    original_settings = world.get_settings()
    settings.fixed_delta_seconds = 0.05
    settings.synchronous_mode = True
    world.apply_settings(settings)
    
    blueprint_library = world.get_blueprint_library()
    actors = []
    
    try:
        map = world.get_map()
        spawn_points = map.get_spawn_points()
        
        # Function to find straight roads
        def find_all_straight_spawns(spawn_points, min_straight_length=150):
            straight_spawns = []
            for sp in spawn_points:
                waypoint = map.get_waypoint(sp.location)
                is_straight = True
                initial_yaw = waypoint.transform.rotation.yaw
                length = 0
                current_wp = waypoint
                
                while length < min_straight_length:
                    next_wps = current_wp.next(5.0)
                    if not next_wps:
                        is_straight = False
                        break
                    next_wp = next_wps[0]
                    yaw_diff = abs(next_wp.transform.rotation.yaw - initial_yaw)
                    if yaw_diff > 3 and yaw_diff < 357:
                        is_straight = False
                        break
                    length += 5.0
                    current_wp = next_wp
                
                if is_straight and length >= min_straight_length:
                    straight_spawns.append(sp)
            return straight_spawns
        
        straight_spawns = find_all_straight_spawns(spawn_points)
        print(f"Found {len(straight_spawns)} straight road segments")
        
        if not straight_spawns:
            print("No straight road found!")
            return
        
        straight_spawn = random.choice(straight_spawns)
        start_location = straight_spawn.location
        
        # Get lane info for clearing
        ego_waypoint = map.get_waypoint(straight_spawn.location)
        lane_id = ego_waypoint.lane_id
        road_id = ego_waypoint.road_id
        
        # Clear the lane of existing vehicles
        print("Clearing lane of any existing vehicles...")
        all_vehicles = world.get_actors().filter('vehicle.*')
        for vehicle in all_vehicles:
            v_loc = vehicle.get_location()
            v_waypoint = map.get_waypoint(v_loc)
            if v_waypoint.road_id == road_id and v_waypoint.lane_id == lane_id:
                dist = math.sqrt(
                    (v_loc.x - start_location.x)**2 + 
                    (v_loc.y - start_location.y)**2
                )
                if dist < 200:
                    vehicle.destroy()
        
        for _ in range(5):
            world.tick()
        
        # Spawn ego vehicle
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_vehicle = world.try_spawn_actor(vehicle_bp, straight_spawn)
        
        if not ego_vehicle:
            print("Failed to spawn ego vehicle!")
            return
        
        actors.append(ego_vehicle)
        print("Ego vehicle spawned")
        
        # Spawn obstacle AFTER 30m zone (random 60-100m ahead)
        obstacle_distance = random.randint(60, 100)
        waypoints_ahead = ego_waypoint.next(float(obstacle_distance))
        
        obstacle_vehicle = None
        if waypoints_ahead:
            obstacle_transform = waypoints_ahead[0].transform
            obstacle_transform.location.z += 0.5
            
            obstacle_types = ['vehicle.audi.a2', 'vehicle.bmw.grandtourer', 'vehicle.citroen.c3']
            obstacle_bp = blueprint_library.find(random.choice(obstacle_types))
            obstacle_vehicle = world.try_spawn_actor(obstacle_bp, obstacle_transform)
            
            if obstacle_vehicle:
                actors.append(obstacle_vehicle)
                obstacle_vehicle.apply_control(carla.VehicleControl(hand_brake=True))
                # Don't print the distance - AEB shouldn't "know" it
                print("Stationary obstacle spawned somewhere ahead (hidden from AEB)")
            else:
                print("Could not spawn obstacle!")
                return
        
        for _ in range(10):
            world.tick()
        
        # Setup RADAR sensor - high sensitivity for early detection
        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', '30')
        radar_bp.set_attribute('vertical_fov', '30')
        radar_bp.set_attribute('range', '150')
        radar_bp.set_attribute('points_per_second', '5000')
        radar_bp.set_attribute('sensor_tick', '0.02')
        
        radar_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        radar = world.spawn_actor(radar_bp, radar_transform, attach_to=ego_vehicle)
        actors.append(radar)
        
        # RADAR data
        radar_data = {'detected': False, 'distance': float('inf')}
        
        def radar_callback(data):
            radar_data['detected'] = False
            radar_data['distance'] = float('inf')
            
            valid_detections = []
            for detection in data:
                azimuth = math.degrees(detection.azimuth)
                altitude = math.degrees(detection.altitude)
                distance = detection.depth
                
                # Only consider obstacles in our lane (narrow azimuth)
                # Wider altitude range to catch vehicles at various heights
                if abs(azimuth) < 8 and -8 < altitude < 20 and distance > 2:
                    valid_detections.append(distance)
            
            if valid_detections:
                radar_data['detected'] = True
                radar_data['distance'] = min(valid_detections)
        
        radar.listen(radar_callback)
        
        # Camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_bp.set_attribute("fov", "100")
        camera_transform = carla.Transform(carla.Location(x=-8, z=4), carla.Rotation(pitch=-15))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        actors.append(camera)
        
        pygame.init()
        display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("AEB - 3 Zone System")
        font = pygame.font.Font(None, 42)
        clock = pygame.time.Clock()
        
        sensor_data = {'image': None}
        
        def camera_callback(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))[:, :, :3][:, :, ::-1]
            sensor_data['image'] = array
        
        camera.listen(camera_callback)
        
        time.sleep(0.5)
        for _ in range(10):
            world.tick()
        
        print("\n" + "=" * 60)
        print("SIMULATION STARTED")
        print("Zone 1: Accelerating for first 30m...")
        print("=" * 60 + "\n")
        
        # Parameters
        ZONE1_DISTANCE = 30.0      # Acceleration zone length
        TARGET_SPEED = 30.0        # Target speed in km/h
        TARGET_STOP_DISTANCE = 4.0 # Aim to stop 4m from obstacle (will coast to ~5-6m)
        MAX_DECEL = 7.0            # Max deceleration m/s²
        
        running = True
        stopped = False
        braking_started = False
        obstacle_first_detected = False
        zone1_complete = False
        max_iterations = 2000  # Safety timeout
        iteration = 0
        
        while running and iteration < max_iterations:
            iteration += 1
            world.tick()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
            
            # Get vehicle state
            velocity = ego_vehicle.get_velocity()
            speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            speed_mps = speed_kmh / 3.6
            
            # Calculate distance traveled from start
            current_loc = ego_vehicle.get_location()
            distance_traveled = math.sqrt(
                (current_loc.x - start_location.x)**2 +
                (current_loc.y - start_location.y)**2
            )
            
            throttle = 0.0
            brake = 0.0
            current_zone = 1
            
            # ============ ZONE 1: ACCELERATION (first 30m) ============
            if distance_traveled < ZONE1_DISTANCE:
                current_zone = 1
                # Pure acceleration - no RADAR checking
                if speed_kmh < TARGET_SPEED:
                    throttle = 0.8
                else:
                    throttle = 0.4
                brake = 0.0
                status = f"ZONE 1: Accelerating ({distance_traveled:.0f}m / {ZONE1_DISTANCE:.0f}m)"
                
            # ============ ZONE 2 & 3: RADAR SCANNING & BRAKING ============
            else:
                if not zone1_complete:
                    zone1_complete = True
                    print(f"Zone 1 complete! Speed: {speed_kmh:.1f} km/h - Now scanning with RADAR...")
                
                # Check RADAR
                if radar_data['detected']:
                    obstacle_dist = radar_data['distance']
                    
                    if not obstacle_first_detected:
                        obstacle_first_detected = True
                        print(f"*** RADAR DETECTED obstacle at {obstacle_dist:.1f}m ***")
                    
                    # ============ ZONE 3: EMERGENCY STOP (within 6m) ============
                    if obstacle_dist <= 6.0:
                        current_zone = 3
                        throttle = 0.0
                        brake = 1.0
                        status = "ZONE 3: STOPPING!"
                        
                    # ============ ZONE 2: GRADUAL BRAKING ============
                    else:
                        current_zone = 2
                        
                        # Calculate required braking based on physics
                        # stopping_distance = v² / (2*a)
                        required_stopping_dist = (speed_mps ** 2) / (2 * MAX_DECEL)
                        safety_margin = 0.5  # Reduced for tighter stopping
                        
                        available_distance = obstacle_dist - TARGET_STOP_DISTANCE
                        
                        if obstacle_dist <= required_stopping_dist + TARGET_STOP_DISTANCE + safety_margin:
                            # Need to brake NOW
                            throttle = 0.0
                            
                            if available_distance > 0 and speed_mps > 0.5:
                                required_decel = (speed_mps ** 2) / (2 * available_distance)
                                # Gradual braking - scale from 20% to 100%
                                brake_ratio = required_decel / MAX_DECEL
                                brake = min(1.0, max(0.2, brake_ratio * 0.9))
                            else:
                                brake = 1.0
                            
                            if not braking_started:
                                braking_started = True
                                print(f"*** BRAKING at {obstacle_dist:.1f}m (speed: {speed_kmh:.1f} km/h) ***")
                            
                            status = f"ZONE 2: Braking {brake:.0%} (dist: {obstacle_dist:.0f}m)"
                        else:
                            # Obstacle detected but far - maintain speed or coast
                            if speed_kmh < TARGET_SPEED:
                                throttle = 0.3
                            else:
                                throttle = 0.0
                            brake = 0.0
                            status = f"ZONE 2: Obstacle at {obstacle_dist:.0f}m - monitoring"
                else:
                    # No obstacle detected - keep driving
                    current_zone = 2
                    if speed_kmh < TARGET_SPEED:
                        throttle = 0.5
                    else:
                        throttle = 0.2
                    brake = 0.0
                    status = "ZONE 2: Scanning - No obstacle"
            
            # Check if stopped
            if speed_kmh < 0.5 and braking_started:
                if not stopped:
                    stopped = True
                    ego_loc = ego_vehicle.get_location()
                    obs_loc = obstacle_vehicle.get_location()
                    actual_dist = math.sqrt(
                        (ego_loc.x - obs_loc.x)**2 + 
                        (ego_loc.y - obs_loc.y)**2
                    )
                    print(f"\n*** STOPPED at {actual_dist:.1f}m from obstacle ***")
                    if actual_dist <= 6.0:
                        print("*** SUCCESS: Stopped within 6m! ***\n")
                    else:
                        print(f"*** Stopped outside 6m range ***\n")
            
            # Apply control
            ego_vehicle.apply_control(carla.VehicleControl(
                throttle=float(throttle),
                brake=float(brake),
                steer=0.0
            ))
            
            # Render
            if sensor_data['image'] is not None:
                surface = pygame.surfarray.make_surface(sensor_data['image'].swapaxes(0, 1))
                display.blit(surface, (0, 0))
                
                y = 15
                # Speed
                display.blit(font.render(f"Speed: {speed_kmh:.1f} km/h", True, (255, 255, 255)), (10, y))
                y += 40
                
                # Distance traveled
                display.blit(font.render(f"Traveled: {distance_traveled:.0f}m", True, (200, 200, 200)), (10, y))
                y += 40
                
                # Zone indicator
                zone_colors = {1: (0, 255, 0), 2: (255, 255, 0), 3: (255, 0, 0)}
                zone_names = {1: "ACCELERATION", 2: "RADAR SCAN", 3: "STOPPING"}
                display.blit(font.render(f"Zone {current_zone}: {zone_names[current_zone]}", True, zone_colors[current_zone]), (10, y))
                y += 40
                
                # RADAR status
                if radar_data['detected']:
                    radar_color = (255, 100, 100)
                    radar_text = f"RADAR: Obstacle at {radar_data['distance']:.0f}m"
                else:
                    radar_color = (100, 255, 100)
                    radar_text = "RADAR: Clear"
                display.blit(font.render(radar_text, True, radar_color), (10, y))
                y += 40
                
                # Status
                display.blit(font.render(status, True, (255, 255, 255)), (10, y))
                
                pygame.display.flip()
            
            clock.tick(20)
            
            if stopped:
                time.sleep(3)
                running = False
        
        print("Simulation complete!")
        
    finally:
        print("Cleaning up...")
        world.apply_settings(original_settings)
        
        for actor in actors:
            if actor is not None and actor.is_alive:
                actor.destroy()
        
        pygame.quit()
        print("Done!")

if __name__ == '__main__':
    main()
