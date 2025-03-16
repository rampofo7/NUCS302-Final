Differentiable MPM Robot Project
This repository contains my work on simulating and optimizing soft-body robots using Taichi and DiffTaichi. The project evolved through several labs, starting with basic simulation setup, moving through complex shape generation and topology (Lab 2), to optimization via gradient-based learning (Lab 3), and finally exploring open-loop control strategies (Lab 4). The optimized version from Lab 3—featuring the “amoeba” robot—is the best-performing model and serves as the core solution for the final project.

Overview

This project uses differentiable physics simulation to design, simulate, and optimize a soft-body robot. The approach is developed incrementally over multiple labs:
Lab 1: Set up the simulation environment using Taichi and run an MPM or rigid body simulation.
Lab 2: Extend the simulation by representing complex geometries and topologies. I experimented with procedural generation to create more interesting robot shapes.
Lab 3: Optimize the robot’s performance using gradient-based methods. A loss function (based on forward distance, speed, and stability) guides iterative updates of actuator control parameters. This version uses an “amoeba” robot design, which proved to be the most stable and efficient.
Lab 4: Test an open-loop control strategy (fixed sine-wave actuation) to compare with the learned controller. The open-loop version does not adapt, and its performance confirms that feedback-based learning leads to better locomotion.

Lab Breakdown

Lab 1: Getting Started
Goal: Install Taichi, set up the simulation environment, and verify that basic physics simulations run correctly.
Outcome: Successfully created a simple simulation and a basic robot shape using rigid body

Lab 2: Representing Complex Structures
Goal: Develop methods to represent advanced shapes and topologies.
Outcome: Switched to soft body and built more intricate designs by combining multiple geometric primitives. This lab paved the way for designing a more organic robot.

Lab 3: Optimization of Custom Structures
Goal: Improve the robot’s locomotion by using gradient-based optimization.
Outcome: Implemented a loss function that encourages forward motion and penalizes instability. The simulation uses automatic differentiation to update control parameters (weights and biases). The “amoeba” function was created in this lab to generate a stable, hollow circular robot that performed best.

Lab 4: Actuation and Control
Goal: Explore a fixed, open-loop control strategy.
Outcome: Implemented a simple sine-wave actuation scheme. Although this approach is easier to implement, it lacks the adaptability seen in Lab 3, as the robot sometimes rolls backward and overall performance is inferior to the optimized version.

Detailed Code Explanation

Global Parameters and Field Definitions
The code begins by defining global parameters that control simulation resolution, material properties, and time-stepping (e.g., n_particles, n_grid, dx, dt, and material constants like E, mu, and la). Using lambda functions, Taichi fields are created for scalars, vectors, and matrices to represent particle positions (x), velocities (v), grid values, and actuation signals.

Field Allocation
The allocate_fields() function sets up the data structure for particles and grid nodes using ti.root.dense(). It allocates arrays for positions, velocities, deformation matrices, and actuation values, and prepares these fields for automatic differentiation by calling ti.root.lazy_grad().

Simulation Kernels
Several kernels drive the simulation:
clear_grid() / clear_particle_grad() / clear_actuation_grad(): Reset the grid and gradient fields at each simulation step.
p2g(f): Transfers particle data to grid nodes. It calculates grid indices using interpolation weights (based on a quadratic kernel) and computes the deformation gradient and stresses.
grid_op(): Applies gravity, friction, and boundary conditions to grid nodes.
g2p(f): Transfers updated grid data back to the particles, updating their velocities and positions.
compute_actuation(t): In Lab 3, this kernel computes the actuator signals using a combination of learned weights, biases, and a sine function.
compute_x_avg() and compute_loss(): These calculate the average position of the solid particles and a loss metric that guides optimization.

Differentiable Simulation and Optimization
Using Taichi’s automatic differentiation, the simulation wraps the advance(s) kernel (which advances the simulation one time step) with decorators (@ti.ad.grad_replaced and @ti.ad.grad_for). In the main loop (inside main()), a gradient tape is used to compute the loss and its gradients, which are then used to update the actuator parameters via gradient descent (with gradient clipping to maintain stability).

Scene Construction and the Amoeba Function
The Scene class is responsible for constructing the robot. It allows adding rectangular patches via the add_rect() method, setting offsets, and finalizing the scene (which tallies the total number of particles and solid particles).

The Amoeba Function
The amoeba() function is central to Lab 3’s optimized design. It constructs a stable, hollow circular structure by:
Creating the Main Body:
 The function iterates over a grid to place small rectangular patches that form a circular shape. It uses geometric conditions (comparing the squared distance to inner and outer radii) to decide which patches to include.
Adding Wheels/Actuators:
 The function defines positions for multiple actuators (wheels) around the perimeter. For each actuator, it places additional patches in a localized, circular pattern to simulate wheels or active regions.
Setting Actuators:
 Finally, it calls scene.set_n_actuators(8) to assign eight unique actuators to the robot.
This design was found to be the most effective during the optimization process in Lab 3, yielding the best balance between forward motion and stability.

Visualization
The visualize(s, folder) function converts the simulation state at time step s into a visual representation. It draws particle positions as circles, colors them based on the actuator’s actuation value, and saves each frame to a designated folder. These frames can later be combined into a GIF or video for analysis.

How to Run
Install Dependencies: pip install taichi numpy matplotlib
Execute the Script:
 Assuming the file is still named diffmpm.py, run: python diffmpm.py --iters 100
 The --iters flag controls the number of optimization iterations. The simulation prints loss values, saves visualization frames (e.g., in

