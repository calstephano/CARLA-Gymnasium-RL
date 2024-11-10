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
        self.camera_img = np.zeros((4, obs_size, obs_size, 3), dtype=np.uint8)     # Placeholder for images from sensors
        self.vehicle = None
       
        # Import camera blueprint library from CARLA
        self.camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

        # Set up camera sensor attributes
        self.camera_bp.set_attribute('image_size_x', str(obs_size))     # Set horizontal resolution
        self.camera_bp.set_attribute('image_size_y', str(obs_size))     # Set vertical resolution
        self.camera_bp.set_attribute('fov', '110')                      # Set field of view
        self.camera_bp.set_attribute('sensor_tick', '0.02')             # Set time (seconds) between sensor captures
        
        # Define transformations for each camera
        self.camera_trans = carla.Transform(carla.Location(x=1.5, z=1.5))
        self.camera_trans2 = carla.Transform(carla.Location(x=0.7, y=0.9, z=1), carla.Rotation(pitch=-35.0, yaw=134.0))
        self.camera_trans3 = carla.Transform(carla.Location(x=0.7, y=-0.9, z=1), carla.Rotation(pitch=-35.0, yaw=-134.0))
        self.camera_trans4 = carla.Transform(carla.Location(x=-1.5, z=1.5), carla.Rotation(yaw=180.0))

    def spawn_and_attach(self, vehicle):
        self.vehicle = vehicle

        # Spawn camera actors
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.vehicle)
        self.camera_sensor2 = self.world.spawn_actor(self.camera_bp, self.camera_trans2, attach_to=self.vehicle)
        self.camera_sensor3 = self.world.spawn_actor(self.camera_bp, self.camera_trans3, attach_to=self.vehicle)
        self.camera_sensor4 = self.world.spawn_actor(self.camera_bp, self.camera_trans4, attach_to=self.vehicle)

        # Listen for data
        self.camera_sensor.listen(lambda data: self._get_camera_img(data, 0))
        self.camera_sensor2.listen(lambda data: self._get_camera_img(data, 1))
        self.camera_sensor3.listen(lambda data: self._get_camera_img(data, 2))
        self.camera_sensor4.listen(lambda data: self._get_camera_img(data, 3))

    def _get_camera_img(self, data, index):
        # Convert camera data to numpy array and store it
        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        array = np.reshape(array, (data.height, data.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.camera_img[index] = array

    def display_camera_img(self, display):
        camera = resize(self.camera_img, (4, self.obs_size, self.obs_size, 3)) * 255
        camera = camera.astype(np.uint8)

        camera_surface = rgb_to_display_surface(camera[0], self.display_size)
        display.blit(camera_surface, (self.display_size * 3, 0))

        camera_surface2 = rgb_to_display_surface(camera[1], self.display_size)
        display.blit(camera_surface2, (self.display_size * 2, 0))

        camera_surface3 = rgb_to_display_surface(camera[2], self.display_size)
        display.blit(camera_surface3, (self.display_size * 4, 0))

        camera_surface4 = rgb_to_display_surface(camera[3], self.display_size)
        display.blit(camera_surface4, (self.display_size * 5, 0))

        return camera


    # TO DO:
    # - Grayscale
    # - Mask 
