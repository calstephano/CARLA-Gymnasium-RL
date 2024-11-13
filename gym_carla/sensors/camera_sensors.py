import carla
import numpy as np
from skimage.transform import resize
from PIL import Image

from gym_carla.envs.misc import *

class CameraSensors:
    def __init__(self, world, obs_size, display_size, color_mode="grayscale"):
        self.world = world
        self.obs_size = obs_size
        self.display_size = display_size
        self.vehicle = None
        self.cameras = []  

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
        self.num_cameras = len(self.camera_positions) 

        # Allow color mode option
        self.color_mode = color_mode.lower()
        if self.color_mode == "rgb":
            # Placeholder for image if RGB
            self.camera_img = np.zeros((self.num_cameras, obs_size, obs_size, 3), dtype=np.uint8)   
        else:
            # Placeholder for image if grayscale
            self.camera_img = np.zeros((self.num_cameras, obs_size, obs_size), dtype=np.uint8)

    def spawn_and_attach(self, vehicle):
        self.vehicle = vehicle
        
        for i, position in enumerate(self.camera_positions):
            # Spawn camera sensor and attach it to the vehicle
            camera = self.world.spawn_actor(self.camera_bp, position, attach_to=self.vehicle)

            # Listen for data
            camera.listen(lambda data, idx=i: self._get_camera_img(data, idx))

            self.cameras.append(camera)

    def _get_camera_img(self, data, index):
        array = np.frombuffer(data.raw_data, dtype=np.uint8)

        if self.color_mode == "rgb":
            array = np.reshape(array, (data.height, data.width, self.num_cameras))[:, :, :3]

        else:
            array = np.reshape(array, (data.height, data.width, self.num_cameras))[:, :, :]

            # Grayscale
            pil_image = Image.fromarray(array)
            grayscale_image = pil_image.convert("L")    # Grayscale = "L" mode
            array = np.array(grayscale_image)

        self.camera_img[index] = array
        
    def display_camera_img(self, display):
        camera = resize(self.camera_img, (self.num_cameras, self.obs_size, self.obs_size, 3)) * 255
        
        camera = camera.astype(np.float32)
        for i in range(self.num_cameras):
            camera_surface = rgb_to_display_surface(camera[i], self.display_size)
            display.blit(camera_surface, (self.display_size * (i + 2), 0))

        return camera
