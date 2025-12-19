# AEB System - BAJA SAEINDIA 2025

Autonomous Emergency Braking (AEB) system for CARLA Simulator.

## Features

- **3-Zone AEB Logic**:
  - Zone 1 (0-100m): Accelerate to 30 km/h
  - Zone 2 (100m+): Cruise and detect obstacles
  - Zone 3 (braking): Gradual braking to stop within 6m ± 2m

- **5 Test Scenarios**:
  - `scenario1_car_stationary.py` - CCRs: Stationary car obstacle
  - `scenario2_car_braking.py` - CCRb: Lead car braking
  - `scenario3_motorcycle_stationary.py` - MCRs: Stationary motorcycle
  - `scenario4_pedestrian_crossing.py` - CPNO: Pedestrian crossing
  - `scenario5_cyclist_crossing.py` - CCNO: Cyclist crossing

## Requirements

- CARLA Simulator 0.9.13+
- Python 3.8+
- pygame, numpy

```bash
pip install -r requirements.txt
```

## Setup CARLA World

Before running scenarios, load the correct map (Town04_Opt has long straight highways):

```python
#!/usr/bin/env python3
"""
CARLA Setup Script - Load Town04_Opt with straight spawn points
"""

import carla

def setup_carla():
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    # Load Town04_Opt (highway map with long straight roads)
    print("Loading Town04_Opt...")
    world = client.load_world('Town04_Opt')
    
    # Verify map
    carla_map = world.get_map()
    print(f"Map: {carla_map.name}")
    
    # Get spawn points
    spawn_points = carla_map.get_spawn_points()
    print(f"Total spawn points: {len(spawn_points)}")
    
    # Known straight spawn points for AEB testing
    # #119 and #121 have 200m+ straight road with <1 degree deviation
    straight_spawns = [119, 121]
    
    print("\nStraight spawn points for AEB testing:")
    for idx in straight_spawns:
        if idx < len(spawn_points):
            sp = spawn_points[idx]
            print(f"  #{idx}: ({sp.location.x:.1f}, {sp.location.y:.1f}) yaw={sp.rotation.yaw:.1f}")
    
    return client, world, spawn_points[119]  # Return spawn point #119

if __name__ == '__main__':
    client, world, spawn_point = setup_carla()
    print(f"\nReady! Use spawn point at ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
```

Or use this one-liner:

```bash
python3 -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(10); w=c.load_world('Town04_Opt'); print('Town04_Opt loaded!')"
```

## Running Scenarios

1. Start CARLA simulator
2. Load the correct map (see above)
3. Run any scenario:

```bash
python3 scenario1_car_stationary.py
python3 scenario2_car_braking.py
python3 scenario3_motorcycle_stationary.py
python3 scenario4_pedestrian_crossing.py
python3 scenario5_cyclist_crossing.py
```

## Test Specification

- **Acceleration Zone**: 100m
- **Obstacle Spawn**: Random 110-150m from start
- **Target Speed**: 30 km/h
- **Stop Distance**: 6m ± 2m (edge-to-edge)
- **Detection**: Starts after 100m traveled (ego doesn't "know" obstacle location beforehand)