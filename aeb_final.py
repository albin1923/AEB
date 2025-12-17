#!/usr/bin/env python3
"""
Simple AEB: Car accelerates, detects obstacle, emergency brakes at 50m from obstacle
Uses direct distance calculation for reliable detection
"""

import carla
import numpy as np
import time
import pygame
import random
import math

def main():
    print("=" * 60)
    print("AEB SYSTEM")
    print("- Car ACCELERATES")
    print("- Emergency brakes when obstacle is within 50m")
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
        # Find STRAIGHT road segments for AEB testing
        map = world.get_map()
        spawn_points = map.get_spawn_points()
        
        # Function to check if road ahead is straight
        def find_all_straight_spawns(spawn_points, min_straight_length=150):
            """Find ALL spawn points with straight road ahead"""
            straight_spawns = []
            
            for sp in spawn_points:
                waypoint = map.get_waypoint(sp.location)
                is_straight = True
                initial_yaw = waypoint.transform.rotation.yaw
                length = 0
                
                # Check waypoints ahead for straightness
                current_wp = waypoint
                while length < min_straight_length:
                    next_wps = current_wp.next(5.0)
                    if not next_wps:
                        is_straight = False
                        break
                    
                    next_wp = next_wps[0]
                    yaw_diff = abs(next_wp.transform.rotation.yaw - initial_yaw)
                    
                    # Allow small yaw differences (< 3 degrees)
                    if yaw_diff > 3 and yaw_diff < 357:
                        is_straight = False
                        break
                    
                    length += 5.0
                    current_wp = next_wp
                
                if is_straight and length >= min_straight_length:
                    straight_spawns.append(sp)
            
            return straight_spawns
        
        # Find ALL straight road spawn points
        straight_spawns = find_all_straight_spawns(spawn_points)
        print(f"Found {len(straight_spawns)} straight road segments")
        
        if not straight_spawns:
            print("No straight road found, using random spawn...")
            random.shuffle(spawn_points)
            straight_spawn = spawn_points[0]
        else:
            # Randomly select one of the straight spawn points
            straight_spawn = random.choice(straight_spawns)
            print("Randomly selected a straight road segment!")
        
        # Spawn ego vehicle on straight road
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_vehicle = world.try_spawn_actor(vehicle_bp, straight_spawn)
        
        if not ego_vehicle:
            print("Failed to spawn ego vehicle!")
            return
        
        ego_spawn = straight_spawn
        actors.append(ego_vehicle)
        print("Ego vehicle spawned on straight road")
        
        # Spawn stationary obstacle at RANDOM distance (50-120m) ahead on the same straight road
        obstacle_distance = random.randint(50, 120)
        print(f"Obstacle will be spawned {obstacle_distance}m ahead")
        
        ego_waypoint = map.get_waypoint(ego_spawn.location)
        waypoints_ahead = ego_waypoint.next(float(obstacle_distance))
        
        obstacle_vehicle = None
        if waypoints_ahead:
            obstacle_transform = waypoints_ahead[0].transform
            obstacle_transform.location.z += 0.5
            
            # Randomly select obstacle vehicle type
            obstacle_types = [
                'vehicle.audi.a2',
                'vehicle.tesla.model3',
                'vehicle.bmw.grandtourer',
                'vehicle.citroen.c3',
                'vehicle.ford.mustang'
            ]
            obstacle_type = random.choice(obstacle_types)
            obstacle_bp = blueprint_library.find(obstacle_type)
            obstacle_vehicle = world.try_spawn_actor(obstacle_bp, obstacle_transform)
            
            if obstacle_vehicle:
                actors.append(obstacle_vehicle)
                obstacle_vehicle.apply_control(carla.VehicleControl(hand_brake=True))
                print(f"Stationary {obstacle_type.split('.')[-1]} spawned {obstacle_distance}m ahead")
            else:
                print("Could not spawn obstacle!")
                return
        
        for _ in range(10):
            world.tick()
        
        # Camera - back POV
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_bp.set_attribute("fov", "100")
        camera_transform = carla.Transform(carla.Location(x=-8, z=4), carla.Rotation(pitch=-15))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        actors.append(camera)
        
        pygame.init()
        display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("AEB - Random Spawn Test")
        font = pygame.font.Font(None, 44)
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
        print("3-ZONE AEB SIMULATION")
        print("Zone 1: Accelerate to 30 ± 3 km/h (27-33 km/h)")
        print("Zone 2: RADAR scanning, gradual braking")
        print("Zone 3: Stop within 6m of obstacle")
        print("=" * 60 + "\n")
        
        # Parameters
        TARGET_SPEED = 30.0          # Target speed: 30 km/h
        SPEED_TOLERANCE = 3.0        # ± 3 km/h tolerance
        MIN_SPEED = TARGET_SPEED - SPEED_TOLERANCE  # 27 km/h
        MAX_SPEED = TARGET_SPEED + SPEED_TOLERANCE  # 33 km/h
        TARGET_STOP_DISTANCE = 5.0   # Target stop at 5m from obstacle
        MAX_DECEL = 8.0              # Max comfortable deceleration in m/s²
        
        # Zone tracking
        current_zone = 1
        zone1_complete = False
        zone1_speed_reached = False
        braking_started = False
        stopped = False
        running = True
        
        # Store start position
        start_loc = ego_vehicle.get_location()
        
        print("ZONE 1: Accelerating to 30 ± 3 km/h...")
        
        while running:
            world.tick()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
            
            # Get vehicle state
            velocity = ego_vehicle.get_velocity()
            speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            speed_mps = speed_kmh / 3.6
            
            # Calculate distance traveled from start
            ego_loc = ego_vehicle.get_location()
            distance_traveled = math.sqrt(
                (ego_loc.x - start_loc.x)**2 + 
                (ego_loc.y - start_loc.y)**2
            )
            
            # Calculate distance to obstacle
            obs_loc = obstacle_vehicle.get_location()
            obstacle_dist = math.sqrt(
                (ego_loc.x - obs_loc.x)**2 + 
                (ego_loc.y - obs_loc.y)**2
            )
            
            throttle = 0.0
            brake = 0.0
            status = ""
            
            # ============== ZONE 1: ACCELERATION ==============
            # Must reach 30 ± 3 km/h before transitioning to Zone 2
            if current_zone == 1:
                # Check if we've reached target speed (27-33 km/h)
                if speed_kmh >= MIN_SPEED and speed_kmh <= MAX_SPEED:
                    if not zone1_speed_reached:
                        zone1_speed_reached = True
                        print(f"*** ZONE 1 COMPLETE: Reached {speed_kmh:.1f} km/h after {distance_traveled:.1f}m ***")
                        print("ZONE 2: RADAR scanning for obstacles...")
                        current_zone = 2
                
                # Accelerate aggressively to reach target speed
                if speed_kmh < MIN_SPEED:
                    throttle = 0.8  # Strong acceleration
                elif speed_kmh < TARGET_SPEED:
                    throttle = 0.6  # Moderate acceleration
                else:
                    throttle = 0.3  # Maintain speed
                
                brake = 0.0
                status = f"ZONE 1: ACCEL ({speed_kmh:.1f} km/h)"
            
            # ============== ZONE 2: RADAR SCANNING & GRADUAL BRAKING ==============
            elif current_zone == 2:
                # Calculate physics-based braking distance
                # d = v² / (2 * a) + safety margin
                stopping_distance = (speed_mps ** 2) / (2 * MAX_DECEL) + TARGET_STOP_DISTANCE + 2.5
                available_distance = obstacle_dist - TARGET_STOP_DISTANCE
                
                if obstacle_dist <= TARGET_STOP_DISTANCE + 1:
                    # Transition to Zone 3 - Emergency stop
                    current_zone = 3
                    throttle = 0.0
                    brake = 1.0
                    status = "ZONE 3: EMERGENCY STOP!"
                    
                elif obstacle_dist <= stopping_distance:
                    # Need to start braking - gradual deceleration
                    throttle = 0.0
                    
                    # Calculate required deceleration for smooth stop
                    if available_distance > 0 and speed_mps > 0.5:
                        required_decel = (speed_mps ** 2) / (2 * available_distance)
                        # Gradual braking - scale brake force with boost for precision
                        brake = (required_decel / MAX_DECEL) * 1.3  # 30% boost
                        brake = min(1.0, max(0.4, brake))
                    else:
                        brake = 1.0
                    
                    status = f"ZONE 2: BRAKING ({brake:.0%})"
                    
                    if not braking_started:
                        braking_started = True
                        print(f"*** RADAR DETECTED obstacle at {obstacle_dist:.1f}m - BRAKING (speed: {speed_kmh:.1f} km/h) ***")
                else:
                    # Maintain speed while scanning
                    if speed_kmh < MIN_SPEED:
                        throttle = 0.5
                    elif speed_kmh > MAX_SPEED:
                        throttle = 0.1
                    else:
                        throttle = 0.3  # Maintain ~30 km/h
                    brake = 0.0
                    status = f"ZONE 2: SCANNING ({obstacle_dist:.0f}m ahead)"
            
            # ============== ZONE 3: STOP ==============
            elif current_zone == 3:
                throttle = 0.0
                brake = 1.0
                status = "ZONE 3: STOPPED"
            
            # Check if stopped
            if speed_kmh < 0.5 and braking_started:
                if not stopped:
                    stopped = True
                    current_zone = 3
                    print(f"\n*** STOPPED at {obstacle_dist:.1f}m from obstacle ***")
                    if obstacle_dist <= 6.0:
                        print("*** SUCCESS: Stopped within 6m! ***")
                    else:
                        print(f"*** Stopped outside 6m range ***")
                    print(f"*** Final speed reached in Zone 1: {TARGET_SPEED:.0f} ± {SPEED_TOLERANCE:.0f} km/h ***\n")
            
            ego_vehicle.apply_control(carla.VehicleControl(
                throttle=float(throttle),
                brake=float(brake),
                steer=0.0
            ))
            
            # Render
            if sensor_data['image'] is not None:
                surface = pygame.surfarray.make_surface(sensor_data['image'].swapaxes(0, 1))
                display.blit(surface, (0, 0))
                
                y = 20
                # Zone indicator
                zone_colors = {1: (0, 255, 0), 2: (255, 255, 0), 3: (255, 0, 0)}
                zone_names = {1: "ZONE 1: ACCELERATING", 2: "ZONE 2: SCANNING", 3: "ZONE 3: STOPPED"}
                display.blit(font.render(zone_names.get(current_zone, ""), True, zone_colors.get(current_zone, (255,255,255))), (10, y))
                y += 45
                
                # Speed with target indicator
                speed_color = (0, 255, 0) if MIN_SPEED <= speed_kmh <= MAX_SPEED else (255, 255, 0)
                display.blit(font.render(f"Speed: {speed_kmh:.1f} km/h (target: {TARGET_SPEED:.0f}±{SPEED_TOLERANCE:.0f})", True, speed_color), (10, y))
                y += 45
                
                # Distance traveled
                display.blit(font.render(f"Distance: {distance_traveled:.1f}m", True, (255, 255, 255)), (10, y))
                y += 45
                
                # Obstacle distance
                color = (255, 0, 0) if braking_started else (0, 255, 0)
                display.blit(font.render(f"Obstacle: {obstacle_dist:.1f}m", True, color), (10, y))
                y += 45
                
                # Status
                status_color = (255, 0, 0) if "BRAKE" in status or "STOP" in status else (0, 255, 0)
                display.blit(font.render(status, True, status_color), (10, y))
                
                pygame.display.flip()
            
            clock.tick(20)
            
            if stopped:
                time.sleep(3)
                running = False
        
        print("Simulation complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("Cleaning up...")
        try:
            camera.stop()
        except:
            pass
        
        for actor in actors:
            try:
                actor.destroy()
            except:
                pass
        
        try:
            world.apply_settings(original_settings)
        except:
            pass
        
        pygame.quit()
        print("Done!")


if __name__ == '__main__':
    main()
