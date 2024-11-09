import carla
import numpy as np

class CameraSensors:
    def __init__(self, world, vehicle, obs_size):
        self.world = world
        self.vehicle = vehicle
        self.obs_size = obs_size
        self.camera_img = np.zeros((4, obs_size, obs_size, 3), dtype=np.uint8)

        # Import blueprint library
        self.camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(obs_size))
        self.camera_bp.set_attribute('image_size_y', str(obs_size))
        self.camera_bp.set_attribute('fov', '110')

        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', '0.02')

        # Define transformations for each camera
        self.camera_trans = carla.Transform(carla.Location(x=1.5, z=1.5))
        self.camera_trans2 = carla.Transform(carla.Location(x=0.7, y=0.9, z=1), carla.Rotation(pitch=-35.0, yaw=134.0))
        self.camera_trans3 = carla.Transform(carla.Location(x=0.7, y=-0.9, z=1), carla.Rotation(pitch=-35.0, yaw=-134.0))
        self.camera_trans4 = carla.Transform(carla.Location(x=-1.5, z=1.5), carla.Rotation(yaw=180.0))

        # Initialize cameras
        self.setup_cameras()

    def setup_cameras(self):
        # Spawn camera actors and listen for data
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.vehicle)
        self.camera_sensor2 = self.world.spawn_actor(self.camera_bp, self.camera_trans2, attach_to=self.vehicle)
        self.camera_sensor3 = self.world.spawn_actor(self.camera_bp, self.camera_trans3, attach_to=self.vehicle)
        self.camera_sensor4 = self.world.spawn_actor(self.camera_bp, self.camera_trans4, attach_to=self.vehicle)

        self.camera_sensor.listen(lambda data: self._process_camera_img(data, 0))
        self.camera_sensor2.listen(lambda data: self._process_camera_img(data, 1))
        self.camera_sensor3.listen(lambda data: self._process_camera_img(data, 2))
        self.camera_sensor4.listen(lambda data: self._process_camera_img(data, 3))

    def _process_camera_img(self, data, index):
        # Convert camera data to numpy array and store it
        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        array = np.reshape(array, (data.height, data.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.camera_img[index] = array
