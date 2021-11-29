---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.1
kernelspec:
  display_name: 'Python 3.10.0 64-bit (''.venv'': venv)'
  language: python
  name: python3
---

```{code-cell} ipython3
from r2 import Constants, VelocityVector
velocity = Constants.LIGHT_VELOCITY * VelocityVector(
        1 / 9,
        1 / 2,
        1 / 4,
)
print(f"The velocity vector is:\n{velocity}\n")
print(f"It's norm is the {velocity.compute_beta()} fraction of light velocity.")
```

```{code-cell} ipython3
from r2 import VelocityQuadriVector
velocity_quad = VelocityQuadriVector.from_space_vector(velocity)
print(f"The velocity quadri-vector built from the previous velocity vector is:\n{velocity_quad}\n")
velocity_quad_proper_ref = velocity_quad.lorentz_transform(velocity)
print(f"This quadri-vector in it's proper reference frame is:\n{velocity_quad_proper_ref}")
```

```{code-cell} ipython3
from r2 import ImpulseQuadriVector
mass = 2
impulse_quad = ImpulseQuadriVector.from_space_vector(velocity_vector=velocity, mass=mass)
print(f"The impulse quadri-vector of a {mass}kg mass with the previous velocity vector is:\n{impulse_quad}\n")
print(f"It's energy is {impulse_quad.compute_energy()}J.\n")
impulse_quad_proper_ref = impulse_quad.lorentz_transform(velocity)
print(f"In it's proper reference frame the same impulse quadri-vector is:\n{impulse_quad_proper_ref}\n")
print(f"In this frame it's energy is {impulse_quad_proper_ref.compute_energy()}J. The reference frame is the frame where the energy is the lowest (no kinetical energy).")
```

```{code-cell} ipython3
from r2 import ImpulseQuadriVector
_lambda = 800e-9
energy = Constants.LIGHT_VELOCITY * Constants.PLANCK_CONSTANT / _lambda
velocity = Constants.LIGHT_VELOCITY * VelocityVector(
        1,
        0,
        0,
)
impulse_quad = ImpulseQuadriVector.from_space_vector(velocity_vector=velocity, energy=energy)
print(f"The impulse quadri-vector of a null mass particule is defined from it's energy. For instance a photon with a {_lambda}m wavelength with the following velocity vector:\n{velocity}\n")
print(f"Has the following impulse quadri-vector:\n{impulse_quad}\n")
```
