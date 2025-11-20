"""
Model class for loading simulation data into Warp arrays.
Handles solid meshes (tetrahedra) only.
"""

import numpy as np
import warp as wp
import igl
from pathlib import Path
from typing import List, Tuple, Optional
import networkx as nx

from .config import SimulationConfig

class State:
    """State storing solver variables that are updated each timestep."""
    
    def __init__(self):
        self.q = None       # Particle positions
        self.qdot = None    # Particle velocities
        self.f_ext = None   # External forces
        self.qt = None      # Previous particle positions
        self.q_tilde = None # Inertia positions

class Model:
    """Model class that loads simulation configuration."""
    
    def __init__(self, config: SimulationConfig, device: str = "cpu", use_networkx_coloring: bool = True):
        """
        Initialize model from simulation configuration.
        
        Args:
            config: Simulation configuration
            device: Warp device ("cpu" or "cuda")
            use_networkx_coloring: If True, use networkx graph coloring for vertex coloring.
                                   If False, use simple single-color approach (debug mode).
        """
        self.config = config
        self.device = device
        self.use_networkx_coloring = use_networkx_coloring
        
        # Device arrays (will be converted to Warp arrays in finalize)
        self.vertices = []  # List of vertex arrays from all objects
        self.tet_indices = []  # List of tetrahedra indices from all objects
        self.initial_velocities = []  # List of initial velocity arrays from all objects
        self.pinned_vertices = []  # List of pinned vertex boolean arrays from all objects
        # TODO: also store the per-element material parameters.
        
        # Host arrays
        self.object_vertex_ranges = []  # (start, end) vertex indices for each object
        self.object_tet_ranges = []  # (start, end) tetrahedra indices for each object

        # Load all objects
        self._load_objects()
        
        # Convert to Warp arrays
        self._finalize()
    
    def _load_objects(self):
        """Load all objects from the configuration."""
        vertex_offset = 0
        tet_offset = 0
        
        for obj_idx, obj_config in enumerate(self.config.objects):
            print(f"Loading object {obj_idx}: {obj_config.mesh}")
            
            # Track vertex range for this object
            vertex_start = vertex_offset
            
            # Only support solid geometry type
            if obj_config.geometry_type != "solid":
                raise ValueError(f"Only 'solid' geometry type is supported. Got: {obj_config.geometry_type}")
            
            # Load tetrahedral mesh
            vertices, tets = self._load_solid_mesh(obj_config.mesh)

            # Normalize mesh if enabled (before applying transform)
            if obj_config.normalize_mesh:
                vertices = vertices / (vertices.max() - vertices.min())
            
            # Apply transform to vertices
            vertices = self._apply_transform(vertices, obj_config.transform)
            
            # Create initial velocities array
            initial_velocities = np.tile(obj_config.initial_velocity, (len(vertices), 1)).astype(np.float32)
            
            # Handle pinned vertices for this object
            is_pinned = np.zeros(len(vertices), dtype=bool)
            if obj_config.pinned_vertices is not None:
                # Check if pinned_vertices is a function
                if callable(obj_config.pinned_vertices):
                    is_pinned = obj_config.pinned_vertices(vertices)
                else:
                    # Otherwise, it's a list of indices
                    for pinned_idx in obj_config.pinned_vertices:
                        is_pinned[pinned_idx] = True
            # Append data
            self.vertices.append(vertices)
            self.tet_indices.append(tets + vertex_offset)
            self.initial_velocities.append(initial_velocities)
            self.pinned_vertices.append(is_pinned)
            
            # Track ranges
            vertex_end = vertex_offset + len(vertices)
            tet_start = tet_offset
            tet_end = tet_offset + len(tets)
            
            # Update offsets
            vertex_offset = vertex_end
            tet_offset = tet_end
            
            # Store ranges for this object
            self.object_vertex_ranges.append((vertex_start, vertex_end))
            self.object_tet_ranges.append((tet_start, tet_end))
    
    def _load_solid_mesh(self, mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load solid mesh (tetrahedra) using igl.readMESH."""
        vertices, tets, _ = igl.readMESH(mesh_path)
        return vertices.astype(np.float32), tets.astype(np.int32)
    
    def _apply_transform(self, vertices: np.ndarray, transform_matrix: List[List[float]]) -> np.ndarray:
        """Apply transform matrix to vertices."""
        # Convert 4x4 matrix to Warp transform
        A = np.array(transform_matrix)
        R = A[:3, :3]
        t = A[:3, 3]
        return vertices @ R + t
    
    def _finalize(self):
        """Convert loaded data to Warp arrays."""
        # Combine all vertices into a single array
        all_vertices = np.vstack(self.vertices)
        
        # Combine all tetrahedra indices
        all_tet_indices = np.vstack(self.tet_indices)
        
        # Combine all initial velocities
        all_initial_velocities = np.vstack(self.initial_velocities)
        
        # Combine all pinned vertices
        all_pinned_vertices = np.hstack(self.pinned_vertices)
        
        # Create Warp arrays
        self.vertices = wp.array(all_vertices, dtype=wp.vec3d, device=self.device)
        self.tet_indices = wp.array(all_tet_indices, dtype=wp.int32, device=self.device)
        self.initial_velocities = wp.array(all_initial_velocities, dtype=wp.vec3d, device=self.device)
        
        # Particle count
        self.particle_count = len(all_vertices)
        
        # Compute tet volumes
        self._compute_tet_volumes()
        
        # Compute material parameters for each tet
        self._compute_material_params()
        
        # Note: Particle masses are computed in the solver initialization (see CD solver)
        # This is done on CPU similar to adjacency computation
        
        # Convert pinned vertices to Warp array
        self.is_pinned = wp.array(all_pinned_vertices, dtype=bool, device=self.device)
        
        # Set gravity
        self.gravity = wp.array([0.0, -9.8, 0.0], dtype=wp.vec3d, device=self.device)
        
        # Initialize contact parameters (stiffness configurable via SimulationConfig)
        self.contact_k = self.config.ground_contact_stiffness

        # Initialize particle coloring (needed for VBD)
        self._initialize_particle_coloring()
    
    def _compute_tet_volumes(self):
        """Compute volumes for each tetrahedron using igl.volume."""
        # Get numpy arrays for computation
        vertices_np = self.vertices.numpy()
        tets_np = self.tet_indices.numpy()
        
        # Use igl to compute volumes for each tet
        volumes = igl.volume(vertices_np, tets_np).astype(np.float64)
        
        # Store as Warp array
        self.tet_volumes = wp.array(volumes, dtype=wp.float64, device=self.device)
    
    def _compute_material_params(self):
        """Compute material parameters (mu, lambda) and density for each tetrahedron from E and nu."""
        # Get material properties from first object (assuming uniform for now)
        material = self.config.objects[0].material
        E = material.youngs
        nu = material.poissons
        density = material.density
        
        # Convert Young's modulus and Poisson's ratio to Lamé parameters
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        
        num_tets = len(self.tet_indices.numpy())
        
        # Create array of (mu, lambda) for each tet
        params_np = np.tile([mu, lam], (num_tets, 1)).astype(np.float64)
        self.material_params = wp.array(params_np, dtype=wp.vec2d, device=self.device)
        
        # Create array of density for each tet
        density_np = np.full(num_tets, density, dtype=np.float64)
        self.tet_density = wp.array(density_np, dtype=wp.float64, device=self.device)
    
    def _initialize_particle_coloring(self):
        """Initialize particle coloring using either networkx or simple approach."""
        if self.use_networkx_coloring:
            self._initialize_networkx_coloring()
        else:
            self._initialize_simple_coloring()
    
    def _initialize_simple_coloring(self):
        """Simple coloring: all particles in one color group (debug mode)."""
        particle_colors_np = np.zeros(self.particle_count, dtype=np.int32)
        self.particle_colors = wp.array(particle_colors_np, dtype=int, device=self.device)
        
        # Create one color group with all particles
        all_particle_ids = np.arange(self.particle_count, dtype=np.int32)
        self.particle_color_groups = [wp.array(all_particle_ids, dtype=int, device=self.device)]
        print(f"Simple coloring: 1 color, {self.particle_count} vertices")
    
    def _initialize_networkx_coloring(self):
        """Use networkx graph coloring for vertex coloring based on element connectivity."""
        # Build vertex adjacency graph from element connectivity
        G = nx.Graph()
        G.add_nodes_from(range(self.particle_count))
        
        # Add edges: vertices connected by a tetrahedron
        tet_indices_np = self.tet_indices.numpy()
        for tet in tet_indices_np:
            # Connect all pairs of vertices in this tetrahedron
            for i in range(4):
                for j in range(i+1, 4):
                    G.add_edge(tet[i], tet[j])
        
        # Perform greedy coloring
        vertex_colors_dict = nx.coloring.greedy_color(G, strategy='smallest_last')
        
        # Convert to numpy array
        particle_colors_np = np.array([vertex_colors_dict[i] for i in range(self.particle_count)], dtype=np.int32)
        self.particle_colors = wp.array(particle_colors_np, dtype=int, device=self.device)
        
        # Group vertices by color
        num_colors = particle_colors_np.max() + 1
        self.particle_color_groups = []
        for color in range(num_colors):
            color_vertex_ids = np.where(particle_colors_np == color)[0].astype(np.int32)
            self.particle_color_groups.append(wp.array(color_vertex_ids, dtype=int, device=self.device))
        
        print(f"Networkx coloring: {num_colors} colors for {self.particle_count} vertices")
        for color, group in enumerate(self.particle_color_groups):
            print(f"  Color {color}: {group.size} vertices")
    
    def state(self) -> State:
        """Clone current model data to a new state."""
        state = State()
        
        state.q = wp.clone(self.vertices)
        state.qdot = wp.clone(self.initial_velocities)
        state.f_ext = wp.zeros(self.particle_count, dtype=wp.vec3d, device=self.device)
        state.qt = wp.clone(self.vertices)
        state.q_tilde = wp.clone(self.vertices)
        
        return state
            
    def print_summary(self):
        """Print a summary of the loaded model."""
        print(f"\n=== Model Summary ===")
        print(f"Device: {self.device}")
        print(f"Total vertices: {len(self.vertices)}")
        print(f"Total tetrahedra: {len(self.tet_indices)}")
        print(f"Total initial velocities: {len(self.initial_velocities)}")
        print(f"Number of objects: {len(self.object_vertex_ranges)}")
        
        for i, obj_config in enumerate(self.config.objects):
            vertex_start, vertex_end = self.object_vertex_ranges[i]
            tet_start, tet_end = self.object_tet_ranges[i]
            print(f"\nObject {i}: {obj_config.mesh}")
            print(f"  Vertex range: {vertex_start}-{vertex_end} ({vertex_end - vertex_start} vertices)")
            print(f"  Tet range: {tet_start}-{tet_end} ({tet_end - tet_start} tetrahedra)")
            print(f"  Material: {obj_config.material.material_model}")
            print(f"  Density: {obj_config.material.density} kg/m³")
            print(f"  Initial velocity: {obj_config.initial_velocity}")
            print(f"  Transform: {obj_config.transform}")
