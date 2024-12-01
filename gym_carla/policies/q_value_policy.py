from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym.spaces import Box
import torch
import torch.nn as nn


class SharedFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that processes the input observation space.
    """
    def __init__(self, observation_space: Box, features_dim: int = 256):
        super(SharedFeatureExtractor, self).__init__(observation_space, features_dim)

        # Example: Shared layers for both camera and state inputs
        self.camera_cnn = nn.Sequential(
            nn.Conv2d(observation_space['camera'].shape[2], 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.state_fc = nn.Sequential(
            nn.Linear(observation_space['state'].shape[0], 64),
            nn.ReLU(),
        )

        # Compute output size of the CNN
        with torch.no_grad():
            sample_input = torch.zeros((1,) + observation_space['camera'].shape)
            n_flatten = self.camera_cnn(sample_input).shape[1]

        self.final_fc = nn.Sequential(
            nn.Linear(n_flatten + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        camera_features = self.camera_cnn(observations['camera'])
        state_features = self.state_fc(observations['state'])
        combined_features = torch.cat((camera_features, state_features), dim=1)
        return self.final_fc(combined_features)


class QValuePolicy(DQNPolicy):
    """
    Custom Q-Value Policy for DQN and DDQN.
    """
    def __init__(self, *args, **kwargs):
        super(QValuePolicy, self).__init__(
            *args,
            features_extractor_class=SharedFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            **kwargs,
        )
