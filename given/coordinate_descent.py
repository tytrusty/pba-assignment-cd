from __future__ import annotations

import warp as wp
from given.adjacency import compute_force_element_adjacency
from assignment.elastic_gradient_and_hessian import elastic_gradient_and_hessian
from assignment.ground_contact_gradient_and_hessian import ground_contact_gradient_and_hessian
from assignment.inertial_gradient_and_hessian import inertial_gradient_and_hessian
from given.lumped_mass import compute_lumped_mass
from given.integrate import integrate_vertex
from given.compute_predicted_positions import compute_predicted_positions
from assignment.solve_coordinates import solve_coordinates
from assignment import dphi_tetdq
from utils import Model, State

wp.set_module_options({"enable_backward": False})

class CoordinateDescentSolver:
    """
    Coordinate Descent solver for implicit time integration.
    
    Uses vertex block coordinate descent (VBD) with graph coloring to solve
    the implicit time integration problem for deformable bodies.
    """

    def __init__(
        self,
        model: Model,
        device: str = "cpu",
        iterations: int = 10,
    ):
        """
        Args:
            model: The `Model` object used to initialize the integrator. Must be identical to the `Model` object passed
                to the `step` function.
            iterations: Number of iterations per step.
        """
        self.model = model
        self.device = device
        self.iterations = iterations
        
        # Compute adjacency information (on CPU)
        self.adjacency = compute_force_element_adjacency(model.tet_indices, model.particle_count).to(self.device)
        
        # Compute lumped mass (on CPU) using tet density array
        lumped_mass, lumped_mass_inv = compute_lumped_mass(
            model.tet_indices,
            model.tet_volumes,
            model.tet_density,
            model.particle_count
        )
        self.lumped_mass = lumped_mass.to(self.device)
        self.lumped_mass_inv = lumped_mass_inv.to(self.device)

        # Initialize dXinv (reference shape matrix inverse) for each tet using a kernel
        num_tets = model.tet_indices.shape[0]
        self.dXinv = wp.zeros(num_tets, dtype=wp.mat((4, 3), dtype=wp.float64), device=self.device)
        
        @wp.kernel
        def compute_dXinv_kernel(
            dXinv: wp.array(dtype=wp.mat((4, 3), dtype=wp.float64)),
            vertices: wp.array(dtype=wp.vec3d),
            tet_indices: wp.array(dtype=wp.int32, ndim=2)
        ):
            tid = wp.tid()
            q0 = vertices[tet_indices[tid, 0]]
            q1 = vertices[tet_indices[tid, 1]]
            q2 = vertices[tet_indices[tid, 2]]
            q3 = vertices[tet_indices[tid, 3]]
            dXinv[tid] = dphi_tetdq(q0, q1, q2, q3)
        
        wp.launch(
            compute_dXinv_kernel,
            dim=num_tets,
            inputs=[self.dXinv, model.vertices.to(self.device), model.tet_indices.to(self.device)],
            device=self.device
        )
        self.gradients = wp.zeros(self.model.particle_count, dtype=wp.vec3d, device=self.device)
        self.hessians = wp.zeros(self.model.particle_count, dtype=wp.mat33d, device=self.device)

    def step(
        self, state_in: State, state_out: State, dt: float
    ):
        wp.launch(
            kernel=compute_predicted_positions,
            inputs=[
                dt,
                self.model.gravity,
                state_in.qt,
                state_in.q,
                state_in.qdot,
                self.lumped_mass_inv,
                state_in.f_ext,
                self.model.is_pinned,
                state_in.q_tilde,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )

        # Coordinate descent iterations (update state_in.q in-place)
        for _iter in range(self.iterations):
            self.gradients.zero_()
            self.hessians.zero_()

            for color in range(len(self.model.particle_color_groups)):
                wp.launch(
                    kernel=inertial_gradient_and_hessian,
                    inputs=[
                        dt,
                        self.model.particle_color_groups[color],
                        self.lumped_mass,
                        state_in.q,
                        state_in.q_tilde,
                        self.model.is_pinned,
                    ],
                    outputs=[self.gradients, self.hessians],
                    dim=self.model.particle_color_groups[color].size,
                    device=self.device,
                )
                
                wp.launch(
                    kernel=ground_contact_gradient_and_hessian,
                    inputs=[
                        state_in.q,
                        self.model.particle_color_groups[color],
                        self.model.contact_k,
                    ],
                    outputs=[self.gradients, self.hessians],
                    dim=self.model.particle_color_groups[color].size,
                    device=self.device,
                )

                # Accumulate elastic forces and Hessians from tetrahedral elements
                wp.launch(
                    kernel=elastic_gradient_and_hessian,
                    inputs=[
                        state_in.q,
                        self.model.particle_color_groups[color],
                        self.adjacency,
                        self.model.tet_indices,
                        self.dXinv,
                        self.model.material_params,
                        self.model.tet_volumes,
                    ],
                    outputs=[self.gradients, self.hessians],
                    dim=self.model.particle_color_groups[color].size,
                    device=self.device,
                )

                # Solve for positions (updates state_in.q in-place for this color)
                wp.launch(
                    kernel=solve_coordinates,
                    inputs=[
                        self.model.particle_color_groups[color],
                        state_in.q,
                        self.model.is_pinned,
                        self.gradients,
                        self.hessians,
                    ],
                    dim=self.model.particle_color_groups[color].size,
                    device=self.device,
                )

        # Integrate: compute final velocity and copy to output state
        wp.launch(
            kernel=integrate_vertex,
            inputs=[dt, state_in.qt, state_in.q],
            outputs=[state_out.q, state_out.qdot],
            dim=self.model.particle_count,
            device=self.device,
        )

