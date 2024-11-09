import carla
import numpy as np
import math

class RadarSensor:
    def __init__(self, world):
        self.world = world
        self.vehicle = None
        self.radar_data = None

        # Import blueprint library from CARLA
        self.radar_bp = world.get_blueprint_library().find('sensor.other.radar')
        
        # Set up radar sensor attributes
        self.radar_bp.set_attribute('horizontal_fov', str(35))                    # Set horizontal field of view angle
        self.radar_bp.set_attribute('vertical_fov', str(20))                      # Set vertical field of view angle
        self.radar_bp.set_attribute('range', str(20))                             # Set detection range (in meters)
        self.radar_bp.set_attribute('points_per_second', '15000')                 # Set radar scan frequency (points per second)
        
        # Set radar location relative to the vehicle
        self.radar_trans = carla.Transform(carla.Location(x=2.0, z=1.0))

    def spawn_and_attach(self, vehicle):
        # Spawn sensor, attach it to the vehicle, and listen for data
        self.vehicle = vehicle
        self.radar_sensor = self.world.spawn_actor(self.radar_bp, self.radar_trans, attach_to=self.vehicle)
        self.radar_sensor.listen(lambda data: self._on_data(data))
    
    def _on_data(self, radar_data):
        # Process radar data and visualize it as per original code
        velocity_range = 7.5  # Max velocity range (m/s)
        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)      # Azimuth angle
            alt = math.degrees(detect.altitude)     # Altitude angle
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)  # Adjust distance for visualization
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            # Normalize velocity for color visualization
            norm_velocity = detect.velocity / velocity_range  # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(-1.0, 0.0, -1.0 - norm_velocity)) * 255.0)

            # Visualize point
            self.world.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

        self.radar_data = radar_data  # Store latest radar data

    def get_data(self):
        return self.radar_data
