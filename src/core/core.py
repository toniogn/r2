from abc import ABC, abstractmethod
from typing import List, Type, Union
import numpy as np
from numpy import double
from copy import deepcopy


class UnconsistentVectorShapeError(ValueError):
    pass


class Constants:
    LIGHT_VELOCITY = 299792458


class Vector(ABC):
    def __init__(self, *args: List[double]) -> None:
        self.coordinates = [arg for arg in args]

    @abstractmethod
    def scalar_product(self, other: "Vector") -> float:
        pass

    def squared_norm(self) -> float:
        return self.scalar_product(self)

    def to_array(self) -> np.ndarray:
        return np.float64([[coordinate] for coordinate in self.coordinates])

    @abstractmethod
    def __add__(self, other: "Vector") -> "Vector":
        return [
            self_coordinate + other_coordinate
            for self_coordinate, other_coordinate in zip(
                self.coordinates, other.coordinates
            )
        ]

    def __mul__(
        self, scalar: Union[int, float, np.float64]
    ) -> Union[
        "Vector",
        "SpaceVector",
        "VelocityVector",
        "QuadriVector",
        "VelocityQuadriVector",
        "ImpulseQuadriVector",
    ]:
        multiplied_self = deepcopy(self)
        multiplied_self.coordinates = [
            scalar * coordinate for coordinate in self.coordinates
        ]
        return multiplied_self

    def __rmul__(
        self, scalar: Union[int, float, np.float64]
    ) -> Union[
        "Vector",
        "SpaceVector",
        "VelocityVector",
        "QuadriVector",
        "VelocityQuadriVector",
        "ImpulseQuadriVector",
    ]:
        return self.__mul__(scalar)

    def __str__(self) -> str:
        return self.to_array().__str__()


class SpaceVector(Vector):
    def __init__(
        self, x_coordinate: float, y_coordinate: float, z_coordinate: float
    ) -> None:
        super().__init__(x_coordinate, y_coordinate, z_coordinate)

    def scalar_product(self, other: Union["SpaceVector", "VelocityVector"]) -> float:
        return sum(
            [
                self_coordinate * other_coordinate
                for self_coordinate, other_coordinate in zip(
                    self.coordinates, other.coordinates
                )
            ]
        )

    def angle(self, other: Union["SpaceVector", "VelocityVector"]) -> float:
        return np.arccos(
            abs(self.scalar_product(other))
            / (self.squared_norm() ** 0.5 * other.squared_norm() ** 0.5)
        )

    def __add__(
        self, other: Union["SpaceVector", "VelocityVector"]
    ) -> Union["SpaceVector", "VelocityVector"]:
        summed_coordinates = super().__add__(other)
        return type(self)(*summed_coordinates)


class VelocityVector(SpaceVector):
    def __init__(
        self, x_coordinate: float, y_coordinate: float, z_coordinate: float
    ) -> None:
        super().__init__(x_coordinate, y_coordinate, z_coordinate)
        self.beta = self.compute_beta()
        self.gamma = self.compute_gamma()

    def compute_beta(self) -> float:
        return self.squared_norm() ** 0.5 / Constants.LIGHT_VELOCITY

    def compute_gamma(self) -> float:
        return 1 / (1 - self.compute_beta() ** 2) ** 0.5


class QuadriVector(Vector):
    def __init__(self, ct_coordinate: float, space_vector: SpaceVector) -> None:
        super().__init__(ct_coordinate, *space_vector.coordinates)

    def scalar_product(self, other: "QuadriVector") -> float:
        return self.coordinates[0] * other.coordinates[0] - SpaceVector(
            *self.coordinates[1:]
        ).scalar_product(SpaceVector(*other.coordinates[1:]))

    def extract_space_vector(self) -> SpaceVector:
        return SpaceVector(*self.coordinates[1:])

    def __add__(
        self,
        other: Union["QuadriVector", "VelocityQuadriVector", "ImpulseQuadriVector"],
    ) -> Union["QuadriVector", "VelocityQuadriVector", "ImpulseQuadriVector"]:
        summed_coordinates = super().__add__(other)
        summed_quadri_vector = deepcopy(self)
        summed_quadri_vector.coordinates = summed_coordinates


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


def lorentz_transform(
    velocity_vector: VelocityVector,
    quadri_vector: Union["QuadriVector", "VelocityQuadriVector", "ImpulseQuadriVector"],
) -> Union["QuadriVector", "VelocityQuadriVector", "ImpulseQuadriVector"]:
    ct_coordinate = quadri_vector.coordinates[0]
    space_vector = quadri_vector.extract_space_vector()
    ct_coordinate_transformed = velocity_vector.gamma * (
        ct_coordinate
        - velocity_vector.scalar_product(space_vector) / Constants.LIGHT_VELOCITY
    )
    space_vector_transformed = (
        space_vector
        + (
            (velocity_vector.gamma - 1)
            * (
                space_vector.scalar_product(velocity_vector)
                / velocity_vector.squared_norm()
            )
            - velocity_vector.gamma * ct_coordinate / Constants.LIGHT_VELOCITY
        )
        * velocity_vector
    )
    quadri_vector_transformed = deepcopy(quadri_vector)
    quadri_vector_transformed.coordinates = [ct_coordinate_transformed, *space_vector_transformed.coordinates]
    return quadri_vector_transformed


if __name__ == "__main__":
    velocity = Constants.LIGHT_VELOCITY * VelocityVector(
        1 / 9,
        1 / 2,
        1 / 4,
    )
    print(velocity.squared_norm() ** 0.5 / Constants.LIGHT_VELOCITY)
    velocity_quad = VelocityQuadriVector(velocity)
    print(velocity_quad)
    velocity_quad_proper_ref = lorentz_transform(velocity, velocity_quad)
    print(velocity_quad_proper_ref)
    impulse_quad = ImpulseQuadriVector(2, velocity)
    print(impulse_quad, impulse_quad.compute_energy())
    impulse_quad_proper_ref = lorentz_transform(velocity, impulse_quad)
    print(impulse_quad_proper_ref, impulse_quad_proper_ref.compute_energy())