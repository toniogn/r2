from abc import ABC, abstractmethod
from typing import Any, List, Type, Union, overload
import numpy as np
from numpy import double
from copy import deepcopy


class UnconsistentVectorShapeError(ValueError):
    pass


class Constants:
    LIGHT_VELOCITY = 299792458


class Vector(ABC):
    def __init__(self, *args: List[double]) -> None:
        self.coordinates: List[double] = args

    def to_array(self) -> np.ndarray:
        return np.float64([[coordinate] for coordinate in self.coordinates])

    def __add__(self, other: "Vector") -> "Vector":
        if isinstance(other, type(self)):
            return type(self)(
                *[
                    sum(coordinates)
                    for coordinates in zip(self.coordinates, other.coordinates)
                ]
            )
        else:
            raise TypeError

    @abstractmethod
    def __mul__(
        self, something: Union[int, float, double, "Vector"]
    ) -> Union["Vector", double]:
        if type(something) in [int, float, double]:
            return Vector(*[something * coordinate for coordinate in self.coordinates])

    def __rmul__(
        self, something: Union[int, float, double, "Vector"]
    ) -> Union["Vector", double]:
        return self.__mul__(something)

    def __str__(self) -> str:
        return self.to_array().__str__()


class SpaceVector(Vector):
    def __init__(
        self, x_coordinate: double, y_coordinate: double, z_coordinate: double
    ) -> None:
        super().__init__(x_coordinate, y_coordinate, z_coordinate)

    def __add__(self, other: Union["SpaceVector", "VelocityVector"]) -> "SpaceVector":
        return super().__add__(other)

    def __mul__(self, something: Union[int, float, double, "SpaceVector", "VelocityVector"]) -> Union["SpaceVector", "VelocityVector", double]:
        super().__mul__(something)
        if isinstance(something, SpaceVector):
            return sum(
                [
                    self_coordinate * something_coordinate
                    for self_coordinate, something_coordinate in zip(
                        self.coordinates, something.coordinates
                    )
                ]
            )
        else:
            raise TypeError

    def angle(self, other: Union["SpaceVector", "VelocityVector"]) -> double:
        return np.arccos(
            abs(self * other) / ((self * self) ** 0.5 * (other * other) ** 0.5)
        )


class VelocityVector(SpaceVector):
    def compute_beta(self) -> float:
        return (self * self) ** 0.5 / Constants.LIGHT_VELOCITY

    def compute_gamma(self) -> float:
        return 1 / (1 - self.compute_beta() ** 2) ** 0.5

    def __add__(self, other: Union["SpaceVector", "VelocityVector"]) -> "VelocityVector":
        return super().__add__(other)


class QuadriVector(Vector):
    def __init__(self, ct_coordinate: float, space_vector: SpaceVector) -> None:
        super().__init__(ct_coordinate, *space_vector.coordinates)

    def __mul__(
        self, something: Union[int, float, double, "QuadriVector", "VelocityQuadriVector", "ImpulseQuadriVector"]
    ) -> Union["QuadriVector", "VelocityQuadriVector", "ImpulseQuadriVector", double]:
        super().__mul__(something)
        if isinstance(something, QuadriVector):
            return (
                self.coordinates[0] * something.coordinates[0]
                - self.extract_space_vector() * something.extract_space_vector()
            )
        else:
            raise TypeError

    def extract_space_vector(self) -> SpaceVector:
        return SpaceVector(*self.coordinates[1:])

    def lorentz_transform(self, velocity_vector: VelocityVector) -> "QuadriVector":
        gamma = velocity_vector.compute_gamma()
        ct_coordinate = self.coordinates[0]
        space_vector = self.extract_space_vector()
        ct_coordinate_transformed = gamma * (
            ct_coordinate - velocity_vector * space_vector / Constants.LIGHT_VELOCITY
        )
        space_vector_transformed = (
            space_vector
            + (
                (gamma - 1)
                * (space_vector * velocity_vector / (velocity_vector * velocity_vector))
                - gamma * ct_coordinate / Constants.LIGHT_VELOCITY
            )
            * velocity_vector
        )
        return QuadriVector(ct_coordinate_transformed, space_vector_transformed)


class VelocityQuadriVector(QuadriVector):
    def __init__(self, vector: Union[VelocityVector, QuadriVector]) -> None:
        if isinstance(vector, VelocityVector):
            gamma = vector.compute_gamma()
            super().__init__(gamma * Constants.LIGHT_VELOCITY, gamma * vector)
        elif isinstance(vector, QuadriVector):
            super().__init__(vector.coordinates[0], vector.extract_space_vector())

    def lorentz_transform(self, velocity_vector: VelocityVector) -> "VelocityQuadriVector":
        return type(self)(super().lorentz_transform(velocity_vector))


class ImpulseQuadriVector(VelocityQuadriVector):
    def __init__(self, mass: float, vector: Union[VelocityVector, QuadriVector]) -> None:
        super().__init__(vector)
        self.coordinates = [mass * coordinate for coordinate in self.coordinates]

    def compute_energy(self):
        return self.coordinates[0] * Constants.LIGHT_VELOCITY

    def lorentz_transform(self, velocity_vector: VelocityVector) -> "ImpulseQuadriVector":
        return type(self)(super().lorentz_transform(velocity_vector))


if __name__ == "__main__":
    velocity = Constants.LIGHT_VELOCITY * VelocityVector(
        1 / 9,
        1 / 2,
        1 / 4,
    )
    print((velocity * velocity) ** 0.5 / Constants.LIGHT_VELOCITY)
    velocity_quad = VelocityQuadriVector(velocity)
    print(velocity_quad)
    velocity_quad_proper_ref = velocity_quad.lorentz_transform(velocity)
    print(velocity_quad_proper_ref)
    impulse_quad = ImpulseQuadriVector(2, velocity)
    print(impulse_quad, impulse_quad.compute_energy())
    impulse_quad_proper_ref = impulse_quad.lorentz_transform(velocity)
    print(impulse_quad_proper_ref, impulse_quad_proper_ref.compute_energy())
