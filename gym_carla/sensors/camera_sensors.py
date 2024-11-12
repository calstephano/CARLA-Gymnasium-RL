import carla
import numpy as np
from skimage.transform import resize
from PIL import Image

from gym_carla.envs.misc import *

class CameraSensors:
    def __init__(self, world, obs_size, display_size):
        self.world = world
        self.obs_size = obs_size
        self.display_size = display_size
        self.cameras = []
        self.camera_img = np.zeros((4, obs_size, obs_size, 3), dtype = np.dtype("uint8"))     # Placeholder for images from sensors
        self.vehicle = None
       
        # Import camera blueprint library from CARLA
        self.camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

        # Set up camera sensor attributes
        self.camera_bp.set_attribute('image_size_x', str(obs_size))     # Set horizontal resolution
        self.camera_bp.set_attribute('image_size_y', str(obs_size))     # Set vertical resolution
        self.camera_bp.set_attribute('fov', '110')                      # Set field of view
        self.camera_bp.set_attribute('sensor_tick', '0.02')             # Set time (seconds) between sensor captures
        
        # Define positions for each camera
        self.camera_positions = [
            carla.Transform(carla.Location(x=1.5, z=1.5)),                                                  # Front view
            carla.Transform(carla.Location(x=0.7, y=0.9, z=1), carla.Rotation(pitch=-35.0, yaw=134.0)),     # Left-back diagonal view
            carla.Transform(carla.Location(x=-1.5, z=1.5), carla.Rotation(yaw=180.0)),                      # Rear view
            carla.Transform(carla.Location(x=0.7, y=-0.9, z=1), carla.Rotation(pitch=-35.0, yaw=-134.0))    # Right-back diagonal view
        ]

    def spawn_and_attach(self, vehicle):
        self.vehicle = vehicle
        
        for i, position in enumerate(self.camera_positions):
            # Spawn camera sensor and attach it to the vehicle
            camera = self.world.spawn_actor(self.camera_bp, position, attach_to=self.vehicle)

            # Listen for data
            camera.listen(lambda data, idx=i: self._get_camera_img(data, idx))

            self.cameras.append(camera)

    def _get_camera_img(self, data, index):
        # Convert camera data to numpy array and store it
        array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
        array = np.reshape(array, (data.height, data.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.camera_img[index] = array

    def display_camera_img(self, display):
        camera = resize(self.camera_img, (4, self.obs_size, self.obs_size, 3)) * 255
        camera = camera.astype(np.float32)

        for i in range(4):
            camera_surface = rgb_to_display_surface(camera[i], self.display_size)
            display.blit(camera_surface, (self.display_size * (i + 2), 0))

        return camera


    # TO DO:
    # - Grayscale
    # - Mask 
