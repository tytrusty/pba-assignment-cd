# PBA Assignment 4: Coordinate Descent for Fast Simulation
In this assignment you will learn to implement coordinate descent for fast simulation. 

**WARNING:** Do not create public repos or forks of this assignment or your solution. Do not post code to your answers online or in the class discussion board. Doing so will result in a 20% deduction from your final grade. 

## Checking out the code and setting up the python environment
These instructions use [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html) for virtual environment. If you do not have it installed, follow the 
instructions at the preceeding link for your operating system

Checkout the code ```git clone git@github.com:dilevin/pba-assignment-fem.git {ROOT_DIR}```, where **{ROOT_DIR}*** is a directory you specify for the source code. 

Next create a virtual environment and install relevant python dependencies install.
```
cd {ROOT_DIR}
conda create -n csc417  python=3.12 -c conda-forge
conda activate csc417
pip install -e . 
```
Optionally, if you have an NVIDIA GPU you might need to install CUDA if you want to use the GPU settings
```
conda install cuda -c nvidia/label/cuda-12.1.0
```
Assignment code templates are stored in the ```{ROOT_DIR}/assginment``` directory. 

**WINDOWS NOTE:** If you want to run the assignments using your GPU you may have to force install torch with CUDA support using 
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Installation without conda
If you are having too many problems with Conda or prefer to use another package manager, we recommend using [UV](https://docs.astral.sh/uv/getting-started/installation/). If you do not have it installed, follow the instructions at the preceeding link for your operating system

Next, create a virtual environment and install relevant python dependencies:

```
cd {ROOT_DIR}
uv venv
uv pip install -e . 
```

If you opted to use UV, you can run the examples using:

```
uv run python main.py <arguments-for-tests>
```


## Tools You Will Use
1. [NVIDIA Warp](https://github.com/NVIDIA/warp) -- python library for kernel programming
2. [PyTorch](https://pytorch.org/) -- python library for array management, deep learning etc ...
   
## Running the Assignment Code
```
cd {ROOT_DIR}
python main.py --scene=tests/{SCENE_PYTHON_FILE}.py
```
By default the assignment code runs on the cpu, you can run it using your GPU via:
```
python main.py --scene=tests/{SCENE_PYTHON_FILE}.py --device=cuda
```
Finally, the code runs, headless and can write results to a USD file which can be viewed in [Blender](https://www.blender.org/):
```
python main.py --scene=tests/{SCENE_PYTHON_FILE}.py --usd_output={FULL_PATH_AND_NAME}.usd --num_steps={Number of steps to run}
```
## Assignment Structure and Instructions
1. You are responsible for implementing all functions found in the [assignments](./assignment) subdirectory.
2. The [tests](./tests) subdirectory contains the scenes, specified as python files,  we will validate your code against.
3. The [test_output](./test_output) subdirectory contains output from the solution code that you can use to validate your code. This output comes in two forms. (1) **USD (Universal Scene Description)** files which contain simulated results. These can be played back in any USD viewer. I use [Blender](https://www.blender.org/). You can output your own simulations as USD files, load both files in blender and examine the simulations side-by-side. (2) Two.pt files which contains the global mass matrix (as a dense matrix) for the one_tet_fall.py scene and the bunny_fall.py scene which you can [load](https://docs.pytorch.org/docs/stable/generated/torch.load.html) and compare your own code to.


# Assignment Overview: Coordinate Descent for Fast Simulation

## Background: From Full Optimization to Coordinate Descent

In the first finite element assignment, at each time step we computed an update to the positions via a minimization over all variables:

$$
\mathbf{q}^{t+1} = \arg\min_{\mathbf{q}} E(\mathbf{q})
$$

This can become quite costly. We typically solve it with Newton's method, with each iteration requiring a linear solve. This approach gives us accurate, stable dynamics, but has high runtime cost.

### Linearly Implicit Time Integration

In graphics, there's been a lot of work to approximate this approach to achieve real-time performance. We often accelerate the optimization by doing **linearly implicit time integration**, which essentially means we do just a single Newton iteration:

$$
\mathbf{q}^{t+1} = \mathbf{q}^t - \mathbf{K}^{-1} \mathbf{g}
$$

where:
- $\mathbf{K}$ is the Hessian: $\mathbf{K} = \frac{\partial^2 E}{\partial \mathbf{q}^2}$
- $\mathbf{g}$ is the gradient: $\mathbf{g} = \frac{\partial E}{\partial \mathbf{q}}$

Even with this approximation, this large, sparse linear solve often still makes it difficult to reach real-time performance.

### The Coordinate Descent Approach

In class, we discussed an alternative approach where we solve for each coordinate of $\mathbf{q} = [\mathbf{x}_0, \ldots, \mathbf{x}_i, \ldots, \mathbf{x}_n]$ independently. Such a local approach is easily mappable to parallel architectures (GPUs), and this strategy is the basis for most "fast solvers" in games.

Solving each coordinate in this way is usually referred to as a type of **coordinate descent** method, and in this assignment we will implement a basic version of this.

Now instead of solving for all coordinates at once, we solve for each coordinate independently:

$$
\mathbf{q}_i^{t+1} = \arg\min_{\mathbf{q}_i} E_i(\mathbf{q})
$$

where $E_i(\mathbf{q})$ is the set of energies tied to the $i$-th coordinate (a coordinate here is just a vertex).

**Important:** While we're solving one coordinate at a time, this energy is still a function of all coordinates. In the case of a tetrahedral mesh, the energy around vertex $i$ is still dependent on the positions of its neighboring vertices (vertices with which it shares a tet).

### Gauss-Seidel and Graph Coloring

**Gauss-Seidel iteration** is a sequential version of this method where when updating vertex $i$, we use the most recent version of all other vertices. This introduces a dependence on prior solutions, but parallelization is still possible via **graph coloring**.

For coloring, we assign each vertex a color such that any two adjacent vertices have different colors. This allows us to parallelize the updates of each color group independently:

1. Assign colors to each vertex
2. For each color group, update the positions of all vertices in the group in parallel

These algorithms are often performed in multiple iterations or "sweeps" through all the vertices. Typically for resource-constrained applications, we just perform a fixed number of sweeps (iterations).

## Algorithm

### Main Time Step Loop

```
Compute predicted positions, q̃
Set initial guess for positions, q = q̃

For iter = 1 to num_iterations:
    For each color j:
        For each vertex i in color j (in parallel):
            q_i = argmin_{q_i} E_i(q)
            
Integrate:
    q^{t+1} = q
    v^{t+1} = (q^{t+1} - q^t) / dt
```

This above loop is what we'll implement in this assignment.

## Energy Functions

Our total energy is the sum of the inertial, elastic, and contact energies:

$$
E(\mathbf{q}) = E^{\text{inertial}}(\mathbf{q}) + E^{\text{elastic}}(\mathbf{q}) + E^{\text{contact}}(\mathbf{q})
$$

### Inertial Energy

The inertial energy for vertex $i$ is:

$$
E_i^{\text{inertial}}(\mathbf{q}) = \frac{1}{2} \frac{m_i}{\Delta t^2} \|\mathbf{q}_i - \tilde{\mathbf{q}}_i\|^2
$$

where:
- $m_i$ is the lumped mass at vertex $i$
- $\tilde{\mathbf{q}}_i$ is the predicted position
- $\Delta t$ is the time step

**Note:** We're not using the full, consistent mass matrix for the inertial energy. Instead, we use a **lumped mass** approximation, where we sum up the mass matrix entries onto the diagonal. This sacrifices momentum coupling, but is a common approximation that makes for easy parallelization. We've computed the lumped matrix for you.

### Elastic Energy

The elastic energy for vertex $i$ is:

$$
E_i^{\text{elastic}}(\mathbf{q}) = \sum_{j \in \mathcal{N}_i} \Psi_j(\mathbf{q}) V_j
$$

where:
- $\mathcal{N}_i$ is the set of elements (tets) adjacent to vertex $i$
- $\Psi_j(\mathbf{q})$ is the energy density of element $j$
- $V_j$ is the volume of element $j$

### Contact Energy

For contact, we implement a simple contact potential for a ground plane:
- Ground plane at $y = 0$ with normal $\mathbf{n} = (0, 1, 0)$
- Contact potential behaves like a spring: as we penetrate the ground, the particle receives a spring force pushing it back above the plane

The contact energy is defined as:

$$
E_i^{\text{contact}}(\mathbf{q}_i) = \begin{cases}
k_c d^2 & \text{if } d < 0 \\
0 & \text{otherwise}
\end{cases}
$$

where:
- $d = \min(0, y_i)$ is the ground penetration magnitude (negative when penetrating)
- $k_c$ is the contact stiffness parameter

When $d \geq 0$ (no penetration), there are no force or Hessian contributions.

## Code Structure

### Provided Code

The algorithm for the coordinate descent loop is provided in [`given/coordinate_descent.py`](given/coordinate_descent.py). This implements the outer loop of the algorithm.

We also provide [`given/adjacency.py`](given/adjacency.py) which contains the `ElementAdjacencyInfo` class. This class essentially gives us $\mathcal{N}_i$ - the set of elements (tets) adjacent to each vertex $i$, which is critical for computing the elastic energy contributions.

#### Adjacency Structure

The `ElementAdjacencyInfo` class provides efficient access to the tets adjacent to each vertex. This is essential for computing elastic energy contributions, as each vertex's energy depends on all tets that contain it.

**Important Functions:**

- `get_vertex_num_adjacent_tets(adjacency, vertex)`: Returns the number of tetrahedra adjacent to a given vertex.
- `get_vertex_adjacent_tet_id_order(adjacency, vertex, i)`: Returns a tuple `(tet_id, vertex_order)` for the i-th adjacent tet, where:
  - `tet_id`: The index of the adjacent tetrahedron
  - `vertex_order`: The local index (0, 1, 2, or 3) of the vertex within that tetrahedron

**Example: Looping Over Adjacent Tets**

Here's how to iterate through all adjacent tets for a vertex:

```python
# Get the number of adjacent tets for vertex i
num_adj_tets = get_vertex_num_adjacent_tets(adjacency, vertex_i)

# Loop through each adjacent tet
for i_adj_tet in range(num_adj_tets):
    # Get the tet ID and the vertex's local order within that tet
    tet_id, vertex_order = get_vertex_adjacent_tet_id_order(adjacency, vertex_i, i_adj_tet)
    
    # Now you can:
    # - Access the tet's vertices: tet_indices[tet_id, 0..3]
    # - Get the tet's dXinv: dXinv[tet_id]
```

This pattern is used in `elastic_gradient_and_hessian.py` to accumulate elastic energy contributions from all adjacent tets for each vertex.

### Code You Need to Write

Much of the code will reuse your implementations from the first finite element assignment.

#### From Assignment 1
- [`assignment/d2neohookean_energy_dF2_tet.py`](assignment/d2neohookean_energy_dF2_tet.py)
- [`assignment/deformation_gradient_tet.py`](assignment/deformation_gradient_tet.py)
- [`assignment/dneohookean_energy_dF_tet.py`](assignment/dneohookean_energy_dF_tet.py)
- [`assignment/dphi_tetdq.py`](assignment/dphi_tetdq.py)
- [`assignment/mass_matrix_tet.py`](assignment/mass_matrix_tet.py)

#### New for This Assignment
- [`assignment/inertial_gradient_and_hessian.py`](assignment/inertial_gradient_and_hessian.py)
- [`assignment/B_tet_vertex.py`](assignment/B_tet_vertex.py)
- [`assignment/elastic_gradient_and_hessian.py`](assignment/elastic_gradient_and_hessian.py)
- [`assignment/ground_contact_gradient_and_hessian.py`](assignment/ground_contact_gradient_and_hessian.py)
- [`assignment/solve_coordinates.py`](assignment/solve_coordinates.py)
