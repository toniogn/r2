import numpy as np
from numpy import dot, transpose


class Constants:
    LIGHT_VELOCITY = 299792458


class SpaceVector(np.array):
    def __init__(
        self, x_coordinate: float, y_coordinate: float, z_coordinate: float
    ) -> None:
        super().__init__([[x_coordinate], [y_coordinate], [z_coordinate]])
        self.beta = self.norm() / Constants.LIGHT_VELOCITY
        self.gamma = 1 / (1 - self.beta ** 2) ** 0.5

    def norm(self) -> float:
        return dot(transpose(self), self)[0][0] ** 0.5


class QuadriVector:
    def __init__(
        self, time: float, x_coordinate: float, y_coordinate: float, z_coordinate: float
    ) -> None:
        self.light_time = Constants.LIGHT_VELOCITY * time
        self.space = SpaceVector(x_coordinate, y_coordinate, z_coordinate)

    def scalar_product(self, second: "QuadriVector") -> float:
        return (
            self.light_time * second.light_time
            - dot(transpose(self.space), second.space)[0][0]
        )


class VelocityQuadriVector(QuadriVector):
    def __init__(self, velocity: SpaceVector) -> None:
        super().__init__(
            velocity.gamma * Constants.LIGHT_VELOCITY,
            velocity.gamma * velocity[0][0],
            velocity.gamma * velocity[1][0],
            velocity.gamma * velocity[2][0],
        )


class ImpulseQuadriVector(VelocityQuadriVector):
    def __init__(self, velocity: SpaceVector, mass: float) -> None:
        super().__init__(mass * velocity)

    def energy(self):
        return self.light_time * Constants.LIGHT_VELOCITY


class LorentzTransform(np.array):
    def __init__(self, velocity: SpaceVector) -> None:
        super().__init__(
            [
                [velocity.gamma, -velocity.gamma * velocity.beta, 0, 0],
                [-velocity.gamma * velocity.beta, velocity.gamma, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
