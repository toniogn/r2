from abc import ABC, abstractmethod
from typing import List, Tuple, overload
import numpy as np
from numpy import double


scalar = float | int | double


class VectorInitializingError(ValueError):
    pass


class Constants:
    LIGHT_VELOCITY = 299792458


class Vector(ABC):
    def __init__(self, *args) -> None:
        self.coordinates: List[scalar] = []
        for arg in args:
            if isinstance(arg, scalar):
                self.coordinates.append(arg)

    def __add__(self, other: "Vector") -> "Vector":
        if len(other.coordinates) == len(self.coordinates):
            return type(self)(
                *[
                    sum(coordinates)
                    for coordinates in zip(self.coordinates, other.coordinates)
                ]
            )
        else:
            raise TypeError

    @abstractmethod
    def __mul__(self, something: scalar) -> "Vector":
        return type(self)(*[something * coordinate for coordinate in self.coordinates])

    def __rmul__(self, something: scalar) -> "Vector":
        return self.__mul__(something)

    def __str__(self) -> str:
        return np.array([[coordinate] for coordinate in self.coordinates]).__str__()


class SpaceVector(Vector):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        if len(self.coordinates) not in [0, 3]:
            raise VectorInitializingError

    @overload
    def __mul__(self, something: "SpaceVector") -> scalar:
        ...

    def __mul__(self, something: scalar) -> "SpaceVector":
        if isinstance(something, scalar):
            return super().__mul__(something)
        elif isinstance(something, SpaceVector):
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

    def __rmul__(self, something: scalar) -> "SpaceVector":
        return super().__rmul__(something)

    def angle(self, other: "SpaceVector") -> scalar:
        return np.arccos(
            abs(self * other) / ((self * self) ** 0.5 * (other * other) ** 0.5)
        )


class VelocityVector(SpaceVector):
    def compute_beta(self) -> float:
        return (self * self) ** 0.5 / Constants.LIGHT_VELOCITY

    def compute_gamma(self) -> float:
        return 1 / (1 - self.compute_beta() ** 2) ** 0.5


class QuadriVector(Vector):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        if len(self.coordinates) not in [0, 4]:
            raise VectorInitializingError

    @overload
    def __mul__(self, something: "QuadriVector") -> scalar:
        ...

    def __mul__(self, something: scalar) -> "QuadriVector":
        if isinstance(something, scalar):
            return super().__mul__(something)
        elif isinstance(something, QuadriVector):
            return (
                self.coordinates[0] * something.coordinates[0]
                - self.to_space_vector() * something.to_space_vector()
            )
        else:
            raise TypeError

    def __rmul__(self, something: scalar) -> "QuadriVector":
        return super().__rmul__(something)

    @classmethod
    def from_space_vector(
        cls, ct_coordinate: scalar, space_vector: SpaceVector
    ) -> "QuadriVector":
        return cls(ct_coordinate, *space_vector.coordinates)

    def to_space_vector(self) -> SpaceVector:
        return SpaceVector(*self.coordinates[1:])

    def lorentz_transform(self, velocity_vector: VelocityVector) -> "QuadriVector":
        gamma = velocity_vector.compute_gamma()
        ct_coordinate = self.coordinates[0]
        space_vector = self.to_space_vector()
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
        return type(self)(
            ct_coordinate_transformed, *space_vector_transformed.coordinates
        )


class VelocityQuadriVector(QuadriVector):
    @classmethod
    def from_space_vector(
        cls, velocity_vector: VelocityVector
    ) -> "VelocityQuadriVector":
        gamma = velocity_vector.compute_gamma()
        return gamma * cls(Constants.LIGHT_VELOCITY, *velocity_vector.coordinates)


class ImpulseQuadriVector(VelocityQuadriVector):
    @classmethod
    def from_space_vector(
        cls, mass: scalar, velocity_vector: VelocityVector
    ) -> "ImpulseQuadriVector":
        gamma = velocity_vector.compute_gamma()
        return (
            mass * gamma * cls(Constants.LIGHT_VELOCITY, *velocity_vector.coordinates)
        )

    def compute_energy(self):
        return self.coordinates[0] * Constants.LIGHT_VELOCITY
    
    def lorentz_transform(
        self, velocity_vector: VelocityVector
    ) -> "ImpulseQuadriVector":
        return super().lorentz_transform(velocity_vector)


if __name__ == "__main__":
    velocity = Constants.LIGHT_VELOCITY * VelocityVector(
        1 / 9,
        1 / 2,
        1 / 4,
    )
    print((velocity * velocity) ** 0.5 / Constants.LIGHT_VELOCITY)
    velocity_quad = VelocityQuadriVector.from_space_vector(velocity)
    print(velocity_quad)
    velocity_quad_proper_ref = velocity_quad.lorentz_transform(velocity)
    print(velocity_quad_proper_ref)
    impulse_quad = ImpulseQuadriVector.from_space_vector(2, velocity)
    print(impulse_quad, impulse_quad.compute_energy())
    impulse_quad_proper_ref = impulse_quad.lorentz_transform(velocity)
    print(impulse_quad_proper_ref, impulse_quad_proper_ref.compute_energy())
