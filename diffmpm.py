import taichi as ti
import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt

real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)

dim = 2
n_particles = 8192
n_solid_particles = 0
n_actuators = 0
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = 1
E = 10
# TODO: update
mu = E
la = E
max_steps = 2048
steps = 1024
gravity = 5 ##INCREASED
target = [0.8, 0.2]

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

actuator_id = ti.field(ti.i32)
particle_type = ti.field(ti.i32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

loss = scalar()

n_sin_waves = 4
weights = scalar()
bias = scalar()
x_avg = vec()

actuation = scalar()
actuation_omega = 35 ##INCREASED
act_strength = 13 ##INCREASED



def allocate_fields():
    ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
    ti.root.dense(ti.i, n_actuators).place(bias)

    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss, x_avg)

    ti.root.lazy_grad()


@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0
        grid_v_in.grad[i, j] = [0, 0]
        grid_m_in.grad[i, j] = 0
        grid_v_out.grad[i, j] = [0, 0]


@ti.kernel
def clear_particle_grad():
    # for all time steps and all particles
    for f, i in x:
        x.grad[f, i] = [0, 0]
        v.grad[f, i] = [0, 0]
        C.grad[f, i] = [[0, 0], [0, 0]]
        F.grad[f, i] = [[0, 0], [0, 0]]


@ti.kernel
def clear_actuation_grad():
    for t, i in actuation:
        actuation[t, i] = 0.0


@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        J = (new_F).determinant()
        if particle_type[p] == 0:  # fluid
            sqrtJ = ti.sqrt(J)
            new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

        F[f + 1, p] = new_F
        r, s = ti.polar_decompose(new_F)

        act_id = actuator_id[p]

        act = actuation[f, ti.max(0, act_id)] * act_strength
        if act_id == -1:
            act = 0.0
        # ti.print(act)

        A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act
        cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        mass = 0.0
        if particle_type[p] == 0:
            mass = 4
            cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * E
        else:
            mass = 1
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                     ti.Matrix.diag(2, la * (J - 1) * J)
        cauchy += new_F @ A @ new_F.transpose()
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[base +
                          offset] += weight * (mass * v[f, p] + affine @ dpos)
                grid_m_in[base + offset] += weight * mass


bound = 3
coeff = 1.0 ##INCREASED FOR MORE FRICTION


@ti.kernel
def grid_op():
    for i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
        if j < bound and v_out[1] < 0:
            v_out[0] = 0
            v_out[1] = 0
            normal = ti.Vector([0.0, 1.0])
            lsq = (normal**2).sum()
            if lsq > 0.5:
                if ti.static(coeff < 0):
                    v_out[0] = 0
                    v_out[1] = 0
                else:
                    lin = v_out.dot(normal)
                    if lin < 0:
                        vit = v_out - lin * normal
                        lit = vit.norm() + 1e-10
                        if lit + coeff * lin <= 0:
                            v_out[0] = 0
                            v_out[1] = 0
                        else:
                            v_out = (1 + coeff * lin / lit) * vit
        if j > n_grid - bound and v_out[1] > 0:
            v_out[0] = 0
            v_out[1] = 0

        grid_v_out[i, j] = v_out


@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C


@ti.kernel
def compute_actuation(t: ti.i32):
    for i in range(n_actuators):
        act = 0.0
        for j in ti.static(range(n_sin_waves)):
            act += weights[i, j] * ti.sin(actuation_omega * t * dt +
                                          2 * math.pi / n_sin_waves * j)
        act += bias[i]
        actuation[t, i] = ti.tanh(act)


@ti.kernel
def compute_x_avg():
    for i in range(n_particles):
        contrib = 0.0
        if particle_type[i] == 1:
            contrib = 1.0 / n_solid_particles
        ti.atomic_add(x_avg[None], contrib * x[steps - 1, i])


@ti.kernel
def compute_loss(): ##COST/FITNESS FUNCTION
    dist = x_avg[None][0]  # Maximize horizontal distance
    speed = v[steps - 1, 0][0]  # Forward speed
    forward_stability = -abs(v[steps - 1, 0][1]) * 0.05  # penalty on vertical motion
    backward_penalty = -min(0, speed) * 0.3  # backward penalty

    # loss function allows for some vertical bounce while focusing on speed distance and forward stability
    loss[None] = -dist - 0.1 * speed + 0.05 * forward_stability + 0.1 * backward_penalty

@ti.ad.grad_replaced
def advance(s):
    clear_grid()
    compute_actuation(s)
    p2g(s)
    grid_op()
    g2p(s)


@ti.ad.grad_for(advance)
def advance_grad(s):
    clear_grid()
    p2g(s)
    grid_op()

    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s)
    compute_actuation.grad(s)


def forward(total_steps=steps):
    # simulation
    for s in range(total_steps - 1):
        advance(s)
    x_avg[None] = [0, 0]
    compute_x_avg()
    compute_loss()


class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0

    def add_rect(self, x, y, w, h, actuation, ptype=1):
        if ptype == 0:
            assert actuation == -1
        global n_particles
        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count
        for i in range(w_count):
            for j in range(h_count):
                self.x.append([
                    x + (i + 0.5) * real_dx + self.offset_x,
                    y + (j + 0.5) * real_dy + self.offset_y
                ])
                self.actuator_id.append(actuation)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)

    def set_offset(self, x, y):
        self.offset_x = x
        self.offset_y = y

    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = n_act


def fish(scene):
    scene.add_rect(0.025, 0.025, 0.95, 0.1, -1, ptype=0)
    scene.add_rect(0.1, 0.2, 0.15, 0.05, -1)
    scene.add_rect(0.1, 0.15, 0.025, 0.05, 0)
    scene.add_rect(0.125, 0.15, 0.025, 0.05, 1)
    scene.add_rect(0.2, 0.15, 0.025, 0.05, 2)
    scene.add_rect(0.225, 0.15, 0.025, 0.05, 3)
    scene.set_n_actuators(4)


def robot(scene):
    scene.set_offset(0.1, 0.03)
    scene.add_rect(0.0, 0.1, 0.3, 0.1, -1)
    scene.add_rect(0.0, 0.0, 0.05, 0.1, 0)
    scene.add_rect(0.05, 0.0, 0.05, 0.1, 1)
    scene.add_rect(0.2, 0.0, 0.05, 0.1, 2)
    scene.add_rect(0.25, 0.0, 0.05, 0.1, 3)
    scene.set_n_actuators(4)

###LAB 3 ROBOT OPTIMIZED.
def amoeba(scene, center_x=0.2, center_y=0.065, outer_radius=0.035, inner_radius=0.016, particle_size=0.01):
    """Creates a stable, hollow circular shape using small square patches."""
    grid_size = int((outer_radius / particle_size) * 4)  # Slightly more points
    grid_size_outer = int((outer_radius / particle_size) * 3.5)

    # Create main body
    for i in range(grid_size):
        for j in range(grid_size):
            x = center_x + (i / grid_size - 0.5) * 2 * outer_radius
            y = center_y + (j / grid_size - 0.5) * 2 * outer_radius

            dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
            outer_sq = outer_radius ** 2
            inner_sq = inner_radius ** 2

            if inner_sq + (particle_size**2) < dist_sq <= outer_sq - (particle_size**2):
                scene.add_rect(x, y, particle_size, particle_size, 1)  # -1 means no actuator

    # Define positions for the four corner wheels and four side wheels
    corner_offset = outer_radius * 0.603
    small_radius = outer_radius * 0.601
    small_inner_radius = small_radius * 0.5
    side_offset = outer_radius * 0.95  

    wheel_positions = [
        (center_x - corner_offset, center_y - corner_offset),  # Bottom-left  (actuator 0)
        (center_x + corner_offset, center_y - corner_offset),  # Bottom-right (actuator 1)
        (center_x - corner_offset, center_y + corner_offset),  # Top-left     (actuator 2)
        (center_x + corner_offset, center_y + corner_offset),  # Top-right    (actuator 3)
        (center_x - side_offset, center_y),  # Left side   (actuator 4)
        (center_x + side_offset, center_y),  # Right side  (actuator 5)
        (center_x, center_y - side_offset),  # Bottom side (actuator 6)
        (center_x, center_y + side_offset)   # Top side    (actuator 7)
    ]

    # Assign a unique actuator to each wheel
    for idx, (cx, cy) in enumerate(wheel_positions):
        for i in range(grid_size_outer):  
            for j in range(grid_size_outer):
                x = cx + (i / grid_size_outer - 0.4) * 2 * small_radius
                y = cy + (j / grid_size_outer - 0.4) * 2 * small_radius

                dist_sq = (x - cx) ** 2 + (y - cy) ** 2
                outer_sq = small_radius ** 2.1
                inner_sq = small_inner_radius ** 2

                if inner_sq < dist_sq <= outer_sq * 0.9:
                    scene.add_rect(x, y, particle_size, particle_size, idx)  # Outer circles denser

    scene.set_n_actuators(8)  # 8 actuators for 8 wheels

gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF)


def visualize(s, folder):
    aid = actuator_id.to_numpy()
    colors = np.empty(shape=n_particles, dtype=np.uint32)
    particles = x.to_numpy()[s]
    actuation_ = actuation.to_numpy()
    for i in range(n_particles):
        color = 0x111111
        if aid[i] != -1:
            act = actuation_[s - 1, int(aid[i])]
            color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
        colors[i] = color
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)

    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=100)
    options = parser.parse_args()

    # initialization
    scene = Scene()
    #robot(scene)
    amoeba(scene)#CALLING ROBOT
    scene.finalize()
    allocate_fields()

    for i in range(n_actuators):
        for j in range(n_sin_waves):
            weights[i, j] = np.random.randn() * 0.01
        bias[i] = np.random.randn() * 0.01  # biases initialized
    for i in range(n_actuators):
        bias[i] += 0.005 


    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]
        v[0, i] = [0, 0]
        if particle_type[i] == 1:  # Only apply to solid particles
            v[0, i] = [0.1, 0.03]  # Increase initial forward push

    losses = []

    for iter in range(options.iters):
        learning_rate = 0.03 * (0.99 ** iter)  # Experiment with this
        with ti.ad.Tape(loss):
            forward()
        l = loss[None]
        losses.append(l)
        print('i=', iter, 'loss=', l)
        #learning_rate = 0.1
        gradient_clip_value = 1.0
        for i in range(n_actuators):
            for j in range(n_sin_waves):
                weights.grad[i, j] = np.clip(weights.grad[i, j], -gradient_clip_value, gradient_clip_value)
                weights[i, j] -= learning_rate * weights.grad[i, j]
            bias.grad[i] = np.clip(bias.grad[i], -gradient_clip_value, gradient_clip_value)
            bias[i] -= learning_rate * bias.grad[i]
                

        if iter % 10 == 0:
            # visualize
            forward(1500)
            for s in range(15, 1500, 16):
                visualize(s, 'diffmpm/iter{:03d}/'.format(iter))


    # ti.profiler_print()
    plt.title("Optimization of Rolling Efficiency")
    plt.ylabel("Loss")
    plt.xlabel("Gradient Descent Iterations")
    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    main()