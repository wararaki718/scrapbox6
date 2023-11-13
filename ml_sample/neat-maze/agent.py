import numpy as np


class Agent:
    def __init__(
        self,
        location,
        heading: float=0,
        radius: float=8.0,
        range_finder_range: float=100.0,
        max_speed: float=3.0,
        max_angular_vel: float=3.0,
        speed_scale: float=1.0,
        angular_scale: float=1.0
    ) -> None:
        self.heading = heading
        self.radius = radius
        self.range_finder_range = range_finder_range
        self.location = location
        self.max_speed = max_speed
        self.max_angular_vel = max_angular_vel
        self.speed_scale = speed_scale
        self.angular_scale = angular_scale

        self.speed = 0
        self.angular_vel = 0

        # defining the range finder sensors
        self.range_finder_angles = np.array([-90.0, -45.0, 0.0, 45.0, 90.0, -180.0])

        # defining the radar sensors
        self.radar_angles = np.array([[315.0, 405.0], [45.0, 135.0], [135.0, 225.0], [225.0, 315.0]])

        # the list to hold range finders activations
        self.range_finders = None
        # the list to hold pie-slice radar activations
        self.radar = None

    def get_obs(self) -> list:
        obs = list(self.range_finders) + list(self.radar)
        return obs

    def apply_control_signals(self, control_signals: np.ndarray) -> None:
        self.angular_vel += (control_signals[0] - 0.5) * self.angular_scale
        self.speed += (control_signals[1] - 0.5) * self.speed_scale

        self.speed = np.clip(self.speed, -self.max_speed, self.max_speed)
        self.angular_vel = np.clip(self.angular_vel, -self.max_angular_vel, self.max_angular_vel)

    def distance_to_exit(self, exit_point):
        return np.linalg.norm(self.location-exit_point)

    def update_rangefinder_sensors(self, walls):

        range_finder_angles = (self.range_finder_angles + self.heading) / 180 * np.pi

        A = np.expand_dims(walls[:,0,:], axis=0)
        B = np.expand_dims(walls[:,1,:], axis=0)

        location = np.expand_dims(self.location, axis=0)
        finder_points = location + self.range_finder_range * np.vstack([np.cos(range_finder_angles), np.sin(range_finder_angles)]).T

        C = np.expand_dims(location, axis=1)
        D = np.expand_dims(finder_points, axis=1)

        AC = A-C
        DC = D-C
        BA = B-A

        rTop = AC[:,:,1] * DC[:,:,0] - AC[:,:,0] * DC[:,:,1]
        sTop = AC[:,:,1] * BA[:,:,0] - AC[:,:,0] * BA[:,:,1]
        Bot = BA[:,:,0] * DC[:,:,1] - BA[:,:,1] * DC[:,:,0]

        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(Bot==0, 0, rTop / Bot)
            s = np.where(Bot==0, 0, sTop / Bot)

        distances = np.where((Bot!=0) & (r>0) & (r<1) & (s>0) & (s<1),
            np.linalg.norm(A + np.expand_dims(r, axis=-1) * BA - C, axis=-1), self.range_finder_range)
        self.range_finders = np.min(distances, axis=1) / self.range_finder_range

    def update_radars(self, exit_point):
        exit_angle = np.arctan2(exit_point[0]-self.location[0], exit_point[1]-self.location[1]) % np.pi
        radar_angles = (self.radar_angles + self.heading) /180 *np.pi

        radar_range = radar_angles[:,1]-radar_angles[:,0]
        radar_diff = (exit_angle-radar_angles[:,0])%(2*np.pi)
        radar = np.zeros(self.radar_angles.shape[0])
        radar[radar_diff<radar_range] = 1
        self.radar = radar
