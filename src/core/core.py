from abc import ABC, abstractmethod
from typing import List, Type, overload
import numpy as np
from numpy import dot
from copy import deepcopy


class HeterogeneousVectorsError(TypeError):
    pass


class Constants:
    LIGHT_VELOCITY = 299792458


class Vector(ABC):
    def __init__(self, *args: List[float]) -> None:
        self.coordinates = args

    @abstractmethod
    def scalar_product(self, other: "Vector") -> float:
        if type(other) != type(self):
            raise HeterogeneousVectorsError
        pass

    def norm(self) -> float:
        return self.scalar_product(self) ** 0.5

    def angle(self, other: "Vector") -> float:
        return np.arccos(abs(self.scalar_product(other)) / (self.norm() * other.norm()))

    def to_array(self) -> np.ndarray:
        return np.array([[coordinate] for coordinate in self.coordinates])

    @overload
    def __mul__(self, scalar: float) -> "VelocityVector":
        pass

    @overload
    def __mul__(self, scalar: float) -> "ImpulseQuadriVector":
        pass

    def __mul__(self, scalar: float) -> "Vector":
        return type(self)(*[scalar * coordinate for coordinate in self.coordinates])

    @overload
    def __rmul__(self, scalar: float) -> "VelocityVector":
        pass

    @overload
    def __rmul__(self, scalar: float) -> "ImpulseQuadriVector":
        pass

    def __rmul__(self, scalar: float) -> "Vector":
        return self.__mul__(scalar)

    def __str__(self) -> str:
        return self.to_array().__str__()


class SpaceVector(Vector):
    def __init__(
        self, first_coordinate: float, second_coordinate: float, third_coordinate: float
    ) -> None:
        super().__init__(first_coordinate, second_coordinate, third_coordinate)

    def scalar_product(self, other: "SpaceVector") -> float:
        return sum(
            [
                self_coordinate * other_coordinate
                for self_coordinate, other_coordinate in zip(
                    self.coordinates, other.coordinates
                )
            ]
        )


class VelocityVector(SpaceVector):
    def __init__(
        self, first_coordinate: float, second_coordinate: float, third_coordinate: float
    ) -> None:
        super().__init__(first_coordinate, second_coordinate, third_coordinate)
        self.beta = self.compute_beta()
        self.gamma = self.compute_gamma()

    def compute_beta(self) -> float:
        return self.norm() / Constants.LIGHT_VELOCITY

    def compute_gamma(self) -> float:
        return 1 / (1 - self.compute_beta() ** 2) ** 0.5


class QuadriVector(Vector):
    def __init__(self, light_time_coordinate: float, space_vector: SpaceVector) -> None:
        super().__init__(light_time_coordinate, *space_vector.coordinates)

    def scalar_product(self, other: "QuadriVector") -> float:
        return self.coordinates[0] * other.coordinates[0] - SpaceVector(
            self.coordinates[1:]
        ).scalar_product(SpaceVector(other.coordinates[1:]))


class VelocityQuadriVector(QuadriVector):
    def __init__(self, velocity_vector: VelocityVector) -> None:
        super().__init__(
            velocity_vector.gamma * Constants.LIGHT_VELOCITY,
            velocity_vector.gamma * velocity_vector,
        )


class ImpulseQuadriVector(VelocityQuadriVector):
    def __init__(self, mass: float, velocity_vector: VelocityVector) -> None:
        super().__init__(velocity_vector)
        self.coordinates = [mass * coordinate for coordinate in self.coordinates]

    def compute_energy(self):
        return self.coordinates[0] * Constants.LIGHT_VELOCITY


class LorentzTransform:
    def __init__(self, velocity: VelocityVector) -> None:
        self.matrix = np.array([
            [velocity.gamma, -velocity.gamma * velocity.beta, 0, 0],
            [-velocity.gamma * velocity.beta, velocity.gamma, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

    @overload
    def transform(self, quadri_vector: ImpulseQuadriVector) -> ImpulseQuadriVector:
        pass

    def transform(self, quadri_vector: QuadriVector) -> QuadriVector:
        transformed_coordinates = dot(self.matrix, quadri_vector.to_array())
        transformed_quadri_vector = deepcopy(quadri_vector)
        transformed_quadri_vector.coordinates = [cell for row in transformed_coordinates for cell in row]
        return transformed_quadri_vector

    def __str__(self) -> str:
        return self.matrix.__str__()


if __name__ == "__main__":
    velocity = Constants.LIGHT_VELOCITY * VelocityVector(
        1 / 2,
        1 / 5,
        1 / 3,
    )
    print(velocity)
    impulse_quad = ImpulseQuadriVector(2, velocity)
    print(impulse_quad)
    lorentz_transform = LorentzTransform(velocity)
    print(lorentz_transform)
    impulse_quad_proper_ref = lorentz_transform.transform(impulse_quad)
    print(impulse_quad_proper_ref)
    print(impulse_quad_proper_ref.compute_energy(), impulse_quad.compute_energy())
    print(impulse_quad.angle(impulse_quad))
