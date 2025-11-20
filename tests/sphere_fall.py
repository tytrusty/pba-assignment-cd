"""
Two tetrahedra scene configuration.
Two tets with one moving and one pinned for testing.
"""

from data import get_data_directory
from utils import SimulationConfig, ObjectConfig, MaterialConfig

class BunnyConfig(SimulationConfig):
    """Two tetrahedra scene configuration."""
    
    def __init__(self):
        # Material for both tets
        tet_material = MaterialConfig(
            density=1000.0,
            material_model="NeoHookean",
            youngs=1e5,  # 100 kPa
            poissons=0.3,
        )
        
        # First tet - moving
        tet1 = ObjectConfig(
            geometry_type="solid",
            mesh=str(get_data_directory() / "sphere.mesh"),
            initial_velocity=[0.0, 0.0, 0.0],  # Moving to the right
            transform=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ],
            material=tet_material
        )
        
        # Initialize simulation configuration
        super().__init__(
            objects=[tet1],
            timestep=0.01,
            gravity=[0.0, -9.8, 0.0],
            ground_contact_stiffness=1e5,
        )
