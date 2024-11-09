# lidar_sensor.py

import carla
import numpy as np

class LIDARSensor:
    def __init__(self, world, vehicle, transform=carla.Transform(carla.Location(x=-0.5, z=1.8))):
        self.world = world
        self.vehicle = vehicle
        self.lidar_data = None

        # Set up LIDAR blueprint
        self.lidar_data = None
        self.lidar_height = 1.8
        self.lidar_trans = carla.Transform(carla.Location(x=-0.5, z=self.lidar_height))
        self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        self.lidar_bp.set_attribute('channels', '64.0')
        self.lidar_bp.set_attribute('range', '100.0')
        self.lidar_bp.set_attribute('upper_fov', '15')
        self.lidar_bp.set_attribute('lower_fov', '-25')
        self.lidar_bp.set_attribute('rotation_frequency', str(1.0 / 0.05))
        self.lidar_bp.set_attribute('points_per_second', '500000')

        # Spawn LIDAR sensor and attach it to the vehicle
        self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, transform, attach_to=self.vehicle)
        self.lidar_sensor.listen(lambda data: self._on_data(data))

    def _on_data(self, data):
        # Convert the point cloud data to a NumPy array and store it
        self.lidar_data = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)

    def get_data(self):
        return self.lidar_data
