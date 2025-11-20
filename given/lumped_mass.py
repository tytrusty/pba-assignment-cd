"""
Lumped mass computation for tetrahedral meshes.
"""

import numpy as np
import warp as wp
from assignment import mass_matrix_tet


@wp.kernel
def accumulate_lumped_mass(
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_volumes: wp.array(dtype=wp.float64),
    tet_density: wp.array(dtype=wp.float64),
    particle_mass: wp.array(dtype=wp.float64),
):
    """
    Accumulate lumped mass for each vertex from adjacent tets.
    
    Uses the consistent mass matrix from mass_matrix_tet and lumps it by summing rows.
    The lumped mass for each vertex is the sum of the corresponding row in the mass matrix.
    Since the mass matrix is block-diagonal with 3x3 blocks per vertex (for x,y,z),
    we only need to sum one row per vertex (they're all the same due to symmetry).
    
    This kernel launches with dim=1 and iterates over all tets sequentially,
    avoiding the need for atomic operations.
    
    Args:
        tet_indices: Array of tet vertex indices [num_tets, 4]
        tet_volumes: Array of tet volumes [num_tets]
        tet_density: Array of tet densities [num_tets]
        particle_mass: Output array for particle masses [num_particles]
    """
    for tet in range(tet_indices.shape[0]):
        # Get the 4 vertices of this tet
        v0 = tet_indices[tet, 0]
        v1 = tet_indices[tet, 1]
        v2 = tet_indices[tet, 2]
        v3 = tet_indices[tet, 3]
        
        # Compute the 12x12 consistent mass matrix for this tet
        M = mass_matrix_tet(tet_density[tet], tet_volumes[tet])
        
        # Lump the mass matrix by summing rows
        # We sum the x-component row for each vertex (rows 0, 3, 6, 9)
        # Due to symmetry, x, y, z components have the same row sum
        mass_v0 = M[0, 0] + M[0, 3] + M[0, 6] + M[0, 9]
        mass_v1 = M[3, 0] + M[3, 3] + M[3, 6] + M[3, 9]
        mass_v2 = M[6, 0] + M[6, 3] + M[6, 6] + M[6, 9]
        mass_v3 = M[9, 0] + M[9, 3] + M[9, 6] + M[9, 9]
        particle_mass[v0] = particle_mass[v0] + mass_v0
        particle_mass[v1] = particle_mass[v1] + mass_v1
        particle_mass[v2] = particle_mass[v2] + mass_v2
        particle_mass[v3] = particle_mass[v3] + mass_v3


def compute_lumped_mass(tet_indices, tet_volumes, tet_density, particle_count):
    """
    Compute lumped mass for all particles from tetrahedral elements.
    
    This function distributes the mass of each tetrahedron equally to its four vertices.
    The mass of each tet is computed as: mass = density * volume.
    
    Args:
        tet_indices: wp.array of tet vertex indices (shape: [num_tets, 4])
        tet_volumes: wp.array of tet volumes (shape: [num_tets])
        tet_density: wp.array of tet densities (shape: [num_tets])
        particle_count: int, number of particles/vertices
        
    Returns:
        tuple: (particle_mass, particle_inv_mass) as wp.arrays
            - particle_mass: Array of particle masses [num_particles]
            - particle_inv_mass: Array of inverse particle masses [num_particles]
    """
    # Move to CPU for computation
    tet_indices = tet_indices.to("cpu")
    tet_volumes = tet_volumes.to("cpu")
    tet_density = tet_density.to("cpu")
    
    with wp.ScopedDevice("cpu"):
        if tet_indices.size:
            # Initialize mass array
            particle_mass = wp.zeros(shape=(particle_count,), dtype=wp.float64)
            
            # Accumulate lumped mass for each vertex
            wp.launch(
                kernel=accumulate_lumped_mass,
                inputs=[tet_indices, tet_volumes, tet_density, particle_mass],
                dim=1,
            )
            
            # Compute inverse mass
            particle_mass_np = particle_mass.numpy()
            particle_inv_mass_np = 1.0 / particle_mass_np            
            particle_inv_mass = wp.array(particle_inv_mass_np, dtype=wp.float64)
            
        else:
            # Empty mesh case
            particle_mass = wp.zeros(shape=(particle_count,), dtype=wp.float64)
            particle_inv_mass = wp.zeros(shape=(particle_count,), dtype=wp.float64)
    
    return particle_mass, particle_inv_mass

