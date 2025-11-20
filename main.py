import polyscope as ps
import polyscope.imgui as psim
import warp as wp
import argparse
import igl
import torch
import numpy as np
from utils import load_config, Model, USDMeshWriter
from given import CoordinateDescentSolver

wp.init()

# Some useful debugging flags https://nvidia.github.io/warp/debugging.html
# wp.config.mode = "debug" 
# wp.config.verify_cuda = True
# wp.config.verify_fp = True

class Simulator:
    """Simulator class that encapsulates all simulation state and methods."""
    
    def __init__(self, config, device):
        """Initialize the simulator with configuration and device.
        
        Args:
            config: Scene configuration
            device: Device to run simulation on ("cpu" or "cuda:0")
        """
        self.device = device
        self.dt = config.timestep
        self.time = 0.0
        self.graph = None
        self.simulating = False
        
        # Create model
        print("Creating model...")
        self.model = Model(config, device=device)
        self.model.print_summary()
        
        # Create solver
        print("Creating solver...")
        self.solver = CoordinateDescentSolver(self.model, iterations=config.iterations, device=device)
        
        # Initialize states
        print("Initializing states...")
        self.state = self.model.state()
        self.state_next = self.model.state()
        
        # Capture CUDA graph for GPU acceleration
        self.capture()
    
    def simulate(self):
        """Internal function that performs a single simulation step."""
        self.solver.step(self.state, self.state_next, self.dt)
        
        # Swap states
        self.state, self.state_next = self.state_next, self.state
    
    def step(self):
        """Run one simulation step using the CoordinateDescentSolver."""
        if self.graph:
            # Use captured CUDA graph for faster execution
            wp.capture_launch(self.graph)
        else:
            # Regular execution
            self.simulate()
        
        self.time += self.dt
    
    def capture(self):
        """Capture the simulation step into a CUDA graph for GPU acceleration."""
        self.graph = None
        if wp.get_device(self.device).is_cuda and False:
            print("Capturing CUDA graph for GPU acceleration...")
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
            print("CUDA graph captured successfully.")
    
    def reset(self):
        """Reset the simulation to initial conditions."""
        print("Resetting Simulation")
        self.state.q.assign(self.model.vertices)
        self.state.qdot.assign(self.model.initial_velocities)
        self.state.f_ext.zero_()
        self.time = 0.0


# Global simulator instance for UI callback
simulator = None


def ui_callback():
    """Polyscope UI callback for interactive simulation."""
    global simulator
    
    # Checkbox to start/stop simulation
    changed_sim, simulator.simulating = psim.Checkbox("Start Simulation", simulator.simulating)
    
    # Reset button
    if psim.Button("Reset Simulation"):
        simulator.reset()
    
    # Run simulation step if active
    if simulator.simulating:
        simulator.step()
        # Update the mesh visualization
        ps.get_surface_mesh("mesh").update_vertex_positions(simulator.state.q.numpy())


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Physics-Based Animation - Coordinate Descent Solver")
    parser.add_argument("--scene", required=True, help="Path to the scene configuration file")
    parser.add_argument("--usd_output", type=str, default=None, help="Output USD file path")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of steps to simulate")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for simulation")
    args = parser.parse_args()
    
    # Set device
    if args.device == "cuda":
        device = "cuda:0"
    else:
        device = "cpu"
    
    # Load configuration
    print(f"Loading scene from {args.scene}")
    config = load_config(args.scene)
    
    # Create simulator
    simulator = Simulator(config, device)
    
    # Compute boundary mesh for visualization/output
    tets_np = simulator.model.tet_indices.numpy()
    triangles_np, _, _ = igl.boundary_facets(tets_np)
    
    if args.usd_output:
        # USD output mode
        if args.num_steps:
            print(f"Writing to USD: {args.usd_output}")
            
            # Convert to torch tensors
            triangles = torch.from_numpy(triangles_np).to(torch.int32)
            vertices = torch.from_numpy(simulator.state.q.numpy()).to(torch.float32)
            
            # Prepare face counts (all triangles)
            face_counts = 3 * torch.ones(triangles.shape[0], dtype=torch.int32)
            
            # Create USD writer
            writer = USDMeshWriter(args.usd_output, fps=1.0/simulator.dt, up_axis="Y", write_velocities=False)
            writer.open(
                face_counts=face_counts.cpu().numpy(),
                face_indices=triangles.reshape(-1).cpu().numpy(),
                num_points=vertices.shape[0]
            )
            
            # Simulate and write frames
            for k in range(args.num_steps):
                print(f"Simulating step {k}/{args.num_steps}")
                
                # Write current frame
                q_np = simulator.state.q.numpy()
                writer.write_frame(q_np, sim_up="Y")
                
                # Simulate one step
                simulator.step()
            
            writer.close()
            print(f"USD output complete. Final time: {simulator.time:.3f}s")
            exit()
        else:
            print("Num steps not provided, skipping USD output")
            exit()
    else:
        # Interactive mode with Polyscope
        print("Starting interactive mode with Polyscope...")
        
        vertices = simulator.state.q.numpy()
        
        # Initialize Polyscope
        ps.init()
        # Enable ground plane if ground contact stiffness is nonzero
        if config.ground_contact_stiffness > 0.0:
            ps.set_ground_plane_mode("tile")
            ps.set_ground_plane_height(0.0) # in world coordinates
        else:
            ps.set_ground_plane_mode("none")
        
        # Register mesh
        ps_mesh = ps.register_surface_mesh("mesh", vertices, triangles_np)
        
        # Set up UI callback
        ps.set_user_callback(ui_callback)
        
        # Show the GUI
        print("Polyscope window opened. Press 'Start Simulation' to begin.")
        ps.show()
