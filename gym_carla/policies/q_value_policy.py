from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import CategoricalDistribution
from torch import nn
import torch as th


class QValueFeatureExtractor(BaseFeaturesExtractor):
    """
    Minimal feature extractor for Dict observation space.
    Processes camera and state inputs separately and combines them.
    """
    def __init__(self, observation_space, features_dim=256):
        super(QValueFeatureExtractor, self).__init__(observation_space, features_dim)

        # Camera CNN (4 cameras with resolution size x size)
        num_cameras = observation_space['camera'].shape[0]
        self.camera_cnn = nn.Sequential(
            nn.Conv2d(num_cameras, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate CNN output size
        with th.no_grad():
            sample_camera = th.zeros((1, num_cameras, observation_space['camera'].shape[1], observation_space['camera'].shape[2]))
            camera_output_dim = self.camera_cnn(sample_camera).shape[1]

        # State fully connected layer (4-dimensional state)
        state_dim = observation_space['state'].shape[0]
        self.state_fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )

        # Combine camera and state features
        self.feature_fc = nn.Sequential(
            nn.Linear(camera_output_dim + 64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Process camera observations through CNN
        camera_features = self.camera_cnn(observations['camera'])

        # Process state observations through fully connected layer
        state_features = self.state_fc(observations['state'])

        # Combine features from camera and state
        combined_features = th.cat([camera_features, state_features], dim=1)

        # Pass through the final feature layer
        return self.feature_fc(combined_features)


class QValuePolicy(BasePolicy):
    """
    Custom DQN policy using QValueFeatureExtractor for Dict observation space.
    """
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(QValuePolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )

        # Custom feature extractor
        self.features_extractor = QValueFeatureExtractor(
            observation_space=observation_space,
            features_dim=256
        )

        # Q-networks (main and target)
        self.q_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_space.n)
        )
        self.q_net_target = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_space.n)
        )

        # Action distribution
        self.action_dist = CategoricalDistribution(action_space.n)

    def forward(self, obs, deterministic=False):
        # Extract features
        features = self.features_extractor(obs)

        # Compute Q-values
        q_values = self.q_net(features)

        # Action probabilities
        actions = self.action_dist.proba_distribution(q_values)
        return actions

    def _predict(self, obs, deterministic=False):
        # Compute Q-values
        q_values = self.q_net(self.features_extractor(obs))

        # Return the best action for deterministic mode
        if deterministic:
            return q_values.argmax(dim=1)
        # Sample an action otherwise
        return self.action_dist.sample()

    def forward_critic(self, obs):
        """
        Returns Q-values for critic computations.
        """
        return self.q_net(self.features_extractor(obs))
