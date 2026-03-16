import torch
from torch import Module
from torch.nn import Conv2d, Sequential, ReLU, Flatten, Dropout2d, Linear, Sigmoid, Softmax


class Network1(Module):
    def __init__(self, alpha):
        super().__init__()

        self.alpha = alpha

        self.backbone = Sequential(
            Conv2d(2, 4, kernel_size=3, stride=1, padding=1, dilation=1),
            ReLU(),
            Conv2d(4, 8, kernel_size=3, stride=1, padding=2, dilation=2),
            ReLU(),
            Dropout2d(0.2),
            Flatten(),
        )

        # output a proximity value in the range [0, 1]
        self.proximity_head = Sequential(
            Linear(8*4*4, 8),
            ReLU(),
            Linear(8, 1),
            Sigmoid(),
        )

        # output a safety value in the range [0, 1]
        self.safety_head = Sequential(
            Linear(8 * 4 * 4, 8),
            ReLU(),
            Linear(8, 1),
            Sigmoid(),
        )

        self.policy_head = Sequential(
            Linear(8 * 4 * 4, 8),
            ReLU(),
            Linear(8, 4),
            Softmax(),
        )


    def forward(self, s):
        # preprocess here?
        x = self.backbone(s)
        p = self.proximity_head(x)
        sigma = self.safety_head(x)

        v = self.alpha * p + (1 - self.alpha) * sigma

        policy_logits = self.policy_head(x)

        return policy_logits, v


class Network2(Module):
    def __init__(self):
        super().__init__()

        self.backbone = Sequential(
            Conv2d(2, 4, kernel_size=3, stride=1, padding=1, dilation=1),
            ReLU(),
            Conv2d(4, 8, kernel_size=3, stride=1, padding=2, dilation=2),
            ReLU(),
            Dropout2d(0.2),
            Flatten(),
        )

        self.policy_head = Sequential(
            Linear(8 * 4 * 4, 8),
            ReLU(),
            Linear(8, 4),
            Softmax(),
        )

        self.value_head = Sequential(
            Linear(2, 8),
            ReLU(),
            Linear(8, 4),
            ReLU(),
            Linear(4, 1),
            Sigmoid(),
        )

    def forward(self, s, p, sigma):
        x = self.backbone(s)
        policy_logits = self.policy_head(x)

        in_features = torch.tensor([p, sigma])

        v = self.value_head(in_features)

        return policy_logits, v


class Network3(Module):
    def __init__(self):
        super().__init__()

        self.feature_combiner = Sequential(
            Linear(2, 8),
            ReLU(),
            Linear(8, 8),
            ReLU(),
        )

        self.value_head = Sequential(
            Linear(8, 4),
            ReLU(),
            Linear(4, 1),
            Sigmoid(),
        )

        self.policy_head = Sequential(
            Linear(8, 4),
            Softmax()
        )

    def forward(self, p, sigma):
        in_features = torch.tensor([p, sigma])
        x = self.feature_combiner(in_features)
        policy_logits = self.policy_head(x)
        v = self.value_head(x)

        return policy_logits, v


class Network4(Module):
    def __init__(self):
        super().__init__()

        self.backbone = Sequential(
            Conv2d(2, 4, kernel_size=3, stride=1, padding=1, dilation=1),
            ReLU(),
            Conv2d(4, 8, kernel_size=3, stride=1, padding=2, dilation=2),
            ReLU(),
            Dropout2d(0.2),
            Flatten(),
        )

        self.proximity_bottleneck = Sequential(
            Linear(8 * 4 * 4, 8),
            ReLU(),
            Linear(8, 4),
            ReLU(),
            Linear(8, 1),
        )

        self.safety_bottleneck = Sequential(
            Linear(8 * 4 * 4, 8),
            ReLU(),
            Linear(8, 4),
            ReLU(),
            Linear(8, 1),
        )

        self.feature_combiner = Sequential(
            ReLU(),
            Linear(2, 8),
            ReLU(),
            Linear(8, 8),
            ReLU(),
        )

        self.policy_head = Sequential(
            Linear(8, 4),
            ReLU(),
            Linear(4, 4),
            Softmax(),
        )

        self.value_head = Sequential(
            Linear(8, 4),
            ReLU(),
            Linear(8, 1),
            Sigmoid(),
        )

    def forward(self, s):
        x = self.backbone(s)

        p = self.proximity_bottleneck(x)
        sigma = self.safety_bottleneck(x)

        in_features = torch.tensor([p, sigma])
        x = self.feature_combiner(in_features)
        policy_logits = self.policy_head(x)
        v = self.value_head(x)

        return policy_logits, v, {'proximity': p, 'sigma': sigma}