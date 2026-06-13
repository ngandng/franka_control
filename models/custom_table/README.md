# Custom Table Model

A simple rectangular table built from primitive boxes.

## Target Outer Size

- Width: 1.60 m (160 cm)
- Depth: 0.80 m (80 cm)
- Height: 0.88 m (88 cm)

## Structural Dimensions

- Tabletop: 1.60 x 0.80 x 0.04 m
- Leg cross-section: 0.05 x 0.05 m
- Leg height: 0.84 m

With this setup, tabletop top surface is exactly at 0.88 m when loaded at z=0.

## Files

- custom_table.urdf: URDF model for PyBullet
- custom_table.sdf: SDF model for Gazebo/ignition
- model.config: model metadata

## PyBullet Example

```python
import pybullet as p
table_id = p.loadURDF(
    "models/custom_table/custom_table.urdf",
    basePosition=[0, 0, 0],
    useFixedBase=True,
)
```
