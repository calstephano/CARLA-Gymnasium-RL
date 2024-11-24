import carla
import numpy as np

class CollisionDetector:
    def __init__(self, world):
        self.world = world
        self.vehicle = None
        self.collision_history = []

        # Import collision blueprint library from CARLA
        try:
            print("[DEBUG] Initializing collision blueprint.")
            self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
            if not self.collision_bp:
                print("[DEBUG] Collision blueprint not found!")
        except Exception as e:
            print(f"[ERROR] Failed to get collision blueprint: {e}")

    def spawn_and_attach(self, vehicle):
        try:
            self.vehicle = vehicle
            print(f"[DEBUG] Attaching collision detector to vehicle ID: {self.vehicle.id}")

            # Spawn the collision detector and attach it to the vehicle
            self.collision_detector = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.vehicle)

            if self.collision_detector:
                print(f"[DEBUG] Collision detector spawned successfully with ID: {self.collision_detector.id}")
            else:
                print("[DEBUG] Collision detector failed to spawn.")

            # Listen for events
            self.collision_detector.listen(lambda event: self._on_collision(event))
            print(f"[DEBUG] Collision detector listener added.")
        except Exception as e:
            print(f"[ERROR] Failed to spawn or attach collision detector: {e}")

    def _on_collision(self, event):
        try:
            print("[DEBUG] Collision event triggered.")
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            print(f"[DEBUG] Collision intensity calculated: {intensity}")

            self.collision_history.append(intensity)
            print(f"[DEBUG] Collision history updated: {self.collision_history}")

            # Keep collision history length to the last recorded collision event
            max_history_length = 1
            if len(self.collision_history) > max_history_length:
                removed_intensity = self.collision_history.pop(0)
                print(f"[DEBUG] Removed old collision intensity: {removed_intensity}")
        except Exception as e:
            print(f"[ERROR] Error during collision event handling: {e}")

    def get_data(self):
        try:
            # Return latest collision data
            print(f"[DEBUG] Getting collision data.")
            return self.collision_data
        except Exception as e:
            print(f"[ERROR] Error getting collision data: {e}")
            return None

    def get_collision_history(self):
        try:
            # Return collision history as a list of intensities
            print(f"[DEBUG] Returning collision history: {self.collision_history}")
            return self.collision_history
        except Exception as e:
            print(f"[ERROR] Error getting collision history: {e}")
            return []

    def get_latest_collision_intensity(self):
        try:
            # Return latest collision intensity if available
            latest_intensity = self.collision_history[-1] if self.collision_history else None
            print(f"[DEBUG] Latest collision intensity: {latest_intensity}")
            return latest_intensity
        except Exception as e:
            print(f"[ERROR] Error getting latest collision intensity: {e}")
            return None

    def clear_collision_history(self):
        try:
            # Clear collision history
            print("[DEBUG] Clearing collision history.")
            self.collision_history = []
        except Exception as e:
            print(f"[ERROR] Error clearing collision history: {e}")
