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
        self.camera_img = np.zeros((4, obs_size, obs_size), dtype = np.dtype("uint8"))     # Placeholder for images from sensors
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
        array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))     # Convert raw data to numpy array
        array = np.reshape(array, (data.height, data.width, 4))[:, :, :3]   # Reshape to a 2D image with RGB channels (discard Alpha)
        
        print(f"RGB image array: {array}\n")

        # Pre-processing: Convert to grayscale
        grayscale_array = np.mean(array, axis=2).astype(np.uint8)

        print(f"Grayscale image array: {grayscale_array}\n")

        self.camera_img[index] = grayscale_array

    def display_camera_img(self, display):
        # Resize the grayscale images for display if needed
        camera = resize(self.camera_img, (4, self.obs_size, self.obs_size, 3)) * 255
        camera = camera.astype(np.uint8)  # Ensure the correct data type for display

        for i in range(4):
            # Create a pygame surface for grayscale image
            camera_surface = pygame.surfarray.make_surface(camera[i])
            # Resize the surface to match the display size
            camera_surface = pygame.transform.scale(camera_surface, (self.display_size, self.display_size))
            # Display the surface
            display.blit(camera_surface, (self.display_size * (i + 2), 0))

        return camera