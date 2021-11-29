R2: Relatively useless toolbox
------------------------------

This package aims to implement some tools and objects of relativity theories in order to observe the relativistic effects on physical cases.


```python
from r2 import Constants, VelocityVector
velocity = Constants.LIGHT_VELOCITY * VelocityVector(
        1 / 9,
        1 / 2,
        1 / 4,
)
print(f"The velocity vector is:\n{velocity}\n")
print(f"It's norm is the {velocity.compute_beta()} fraction of light velocity.")
```

    The velocity vector is:
    [[3.33102731e+07]
     [1.49896229e+08]
     [7.49481145e+07]]
    
    It's norm is the 0.5699523480189775 fraction of light velocity.
    


```python
from r2 import VelocityQuadriVector
velocity_quad = VelocityQuadriVector.from_space_vector(velocity)
print(f"The velocity quadri-vector built from the previous velocity vector is:\n{velocity_quad}\n")
velocity_quad_proper_ref = velocity_quad.lorentz_transform(velocity)
print(f"This quadri-vector in it's proper reference frame is:\n{velocity_quad_proper_ref}")
```

    The velocity quadri-vector built from the previous velocity vector is:
    [[3.64854055e+08]
     [4.05393394e+07]
     [1.82427027e+08]
     [9.12135137e+07]]
    
    This quadri-vector in it's proper reference frame is:
    [[2.99792458e+08]
     [0.00000000e+00]
     [0.00000000e+00]
     [0.00000000e+00]]
    


```python
from r2 import ImpulseQuadriVector
mass = 2
impulse_quad = ImpulseQuadriVector.from_space_vector(velocity_vector=velocity, mass=mass)
print(f"The impulse quadri-vector of a {mass}kg mass with the previous velocity vector is:\n{impulse_quad}\n")
print(f"It's energy is {impulse_quad.compute_energy()}J.\n")
impulse_quad_proper_ref = impulse_quad.lorentz_transform(velocity)
print(f"In it's proper reference frame the same impulse quadri-vector is:\n{impulse_quad_proper_ref}\n")
print(f"In this frame it's energy is {impulse_quad_proper_ref.compute_energy()}J. The reference frame is the frame where the energy is the lowest (no kinetical energy).")
```

    The impulse quadri-vector of a 2kg mass with the previous velocity vector is:
    [[7.29708110e+08]
     [8.10786789e+07]
     [3.64854055e+08]
     [1.82427027e+08]]
    
    It's energy is 2.1876098782138845e+17J.
    
    In it's proper reference frame the same impulse quadri-vector is:
    [[5.99584916e+08]
     [0.00000000e+00]
     [0.00000000e+00]
     [0.00000000e+00]]
    
    In this frame it's energy is 1.7975103574736355e+17J. The reference frame is the frame where the energy is the lowest (no kinetical energy).
    


```python
from r2 import ImpulseQuadriVector
_lambda = 800e-9
energy = Constants.LIGHT_VELOCITY * Constants.PLANCK_CONSTANT / _lambda
photon_velocity = Constants.LIGHT_VELOCITY * VelocityVector(
        1,
        0,
        0,
)
impulse_quad = ImpulseQuadriVector.from_space_vector(velocity_vector=photon_velocity, energy=energy)
print(f"The impulse quadri-vector of a null mass particule is defined from it's energy. For instance a photon with a {_lambda}m wavelength with the following velocity vector:\n{photon_velocity}\n")
print(f"Has the following impulse quadri-vector:\n{impulse_quad}\n")
```

    The impulse quadri-vector of a null mass particule is defined from it's energy. For instance a photon with a 8e-07m wavelength with the following velocity vector:
    [[2.99792458e+08]
     [0.00000000e+00]
     [0.00000000e+00]]
    
    Has the following impulse quadri-vector:
    [[8.28258755e-28]
     [8.28258755e-28]
     [0.00000000e+00]
     [0.00000000e+00]]
    
    

The proper reference frame is not defined for this quadri-vector because by definition the speed of light is absolute. You can't find a reference frame where the photon is not moving in space. But, you can observe the compatibility of the Lorentz Transform with the light speed absolutism:


```python
impulse_quad.lorentz_transform(velocity)
print(impulse_quad)
```

    [[8.28258755e-28]
     [8.28258755e-28]
     [0.00000000e+00]
     [0.00000000e+00]]
    
