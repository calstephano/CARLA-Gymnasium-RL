import carla
import threading
import pygame
import numpy as np
from skimage.transform import resize
from gym_carla.display.display_utils import grayscale_to_display_surface
import cv2

class CameraSensors:
  def __init__(self, world, obs_size, display_size, window_size=5):
    self.world = world
    self.obs_size = obs_size
    self.display_size = display_size
    self.window_size = window_size
    self.cameras = []
    self.camera_img = np.zeros((4, window_size, obs_size, obs_size), dtype=np.uint8)
    self.vehicle = None
    self.lock = threading.Lock()

    # Import camera blueprint library from CARLA
    self.camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

    # Set up camera sensor attributes
    self.camera_bp.set_attribute('image_size_x', str(obs_size))   # Set horizontal resolution
    self.camera_bp.set_attribute('image_size_y', str(obs_size))   # Set vertical resolution
    self.camera_bp.set_attribute('fov', '110')            				# Set field of view
    self.camera_bp.set_attribute('sensor_tick', '0.02')       		# Set time (seconds) between sensor captures

    # Define positions for each camera
    self.camera_positions = [
      carla.Transform(carla.Location(x=1.5, z=1.5)),                          											# Front view
      carla.Transform(carla.Location(x=0.7, y=0.9, z=1), carla.Rotation(pitch=-35.0, yaw=134.0)),   # Left-back diagonal view
      carla.Transform(carla.Location(x=-1.5, z=1.5), carla.Rotation(yaw=180.0)),            				# Rear view
      carla.Transform(carla.Location(x=0.7, y=-0.9, z=1), carla.Rotation(pitch=-35.0, yaw=-134.0))  # Right-back diagonal view
    ]

  def spawn_and_attach(self, vehicle):
    self.vehicle = vehicle

    for i, position in enumerate(self.camera_positions):
      # Spawn camera sensor and attach it to the vehicle
      camera = self.world.spawn_actor(self.camera_bp, position, attach_to=self.vehicle)

      # Listen for data
      camera.listen(lambda data, idx=i: self._get_camera_img(data, idx))

      self.cameras.append(camera)

  def detect_lanes(self, grayscale_image, camera_index):
    """Detect lanes in a given grayscale image, customized for different cameras and taking angle into account."""
    # Apply Canny Edge Detection
    edges = cv2.Canny(grayscale_image, 50, 150)

    # Create a mask to define the region of interest (ROI)
    mask = np.zeros_like(edges)

    # Use a trapezoidal shape to focus on the lanes ahead
    polygon = np.array([[
      (0, grayscale_image.shape[0]),                                       # Bottom-left corner
      (grayscale_image.shape[1] // 4, grayscale_image.shape[0] // 2),      # Top-left part
      (grayscale_image.shape[1] * 3 // 4, grayscale_image.shape[0] // 2),  # Top-right part
      (grayscale_image.shape[1], grayscale_image.shape[0])                 # Bottom-right corner
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)                                         # Apply mask to the region of interest

    # Apply the mask to the edges
    roi_edges = cv2.bitwise_and(edges, mask)

    # Apply Hough Line Transform to detect lane lines
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=50)

    # Filter lines based on angle
    lane_img = grayscale_image.copy()
    if lines is not None:
      for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate the angle of the line
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # Angle in degrees

        # Filter lines based on angle (e.g., keep lines between -45 and 45 degrees for lane markings)
        if -45 < angle < 45:
          cv2.line(lane_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    return lane_img


  def _get_camera_img(self, data, index):
    """Process and store camera sensor data as a preprocessed grayscale image with a sliding window."""
    # Convert raw data to numpy array and reshape to a 2D image with RGB channels
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (data.height, data.width, 4))[:, :, :3]
    
    # Convert to grayscale
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

    # Resize the grayscale image to the observation size
    resized_array = resize(gray, (self.obs_size, self.obs_size), preserve_range=True).astype(np.uint8)

    # Detect lanes in the resized grayscale imag only for front camera
    if index == 0:
        lane_image = self.detect_lanes(resized_array, index)
    else:
        lane_image = resized_array
    
    # Safely update the corresponding index in the cache
    with self.lock:
      # Shift the window for this camera, dropping the oldest image and adding the new one
      self.camera_img[index] = np.roll(self.camera_img[index], shift=-1, axis=0)
      self.camera_img[index, -1] = lane_image

  def render_camera_img(self, display):
    with self.lock:
      for i, camera_position in enumerate(self.camera_positions):
        camera_surface = grayscale_to_display_surface(self.camera_img[i, -1], self.display_size)
        display.blit(camera_surface, (self.display_size * (i + 2), 0))

  def stop(self):
    self.stopped = True
    for camera in self.cameras:
      camera.stop()
    self.cameras = []