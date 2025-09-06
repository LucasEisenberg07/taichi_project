import taichi as ti

ti.init(arch=ti.gpu)

n = 1024
num_circles = 100
dt = 0.01
particle_radius = 15
particle_radius_fraction = particle_radius / n
damping_coefficient = 0.8
mouse_size_factor = 2.0
elevator_fraction = 0.2
elevator_acceleration = 2
gravity = -9.8

positions_circles = ti.Vector.field(2, dtype=ti.f32, shape=num_circles)
velocities_circles = ti.Vector.field(2, dtype=ti.f32, shape=num_circles)
accelerations_circles = ti.Vector.field(2, dtype=ti.f32, shape=num_circles)
colors_circles = ti.field(dtype=ti.f32, shape=num_circles)

gravity = ti.Vector([0, gravity])

mouse_pos = ti.Vector([0.0, 0.0])
previous_mouse_pos = ti.Vector([0.0, 0.0])
mouse_velocity = ti.Vector([0.0, 0.0])


@ti.kernel
def initialize_particles():
    for i in range(num_circles):
        # Randomize initial positions, velocities, and colors
        positions_circles[i] = ti.Vector([ti.random() * 2, ti.random() * 2])
        velocities_circles[i] = ti.Vector([ti.random() * 2, ti.random() * 2]) * 0.5
        colors_circles[i] = ti.random() * 0xffffff

@ti.kernel
def update():
    for i in range(num_circles):
        if positions_circles[i][0] < elevator_fraction:
            accelerations_circles[i] = ti.Vector([0, elevator_acceleration])
        else:
            accelerations_circles[i] = gravity
        velocities_circles[i] += accelerations_circles[i] * dt
        positions_circles[i] += velocities_circles[i] * dt

        for j in range(i):
            if i == j:
                continue
            distance = positions_circles[i] - positions_circles[j]
            if distance.norm() < (2 * particle_radius_fraction):
                normal = distance.normalized()
                relative_velocity = velocities_circles[i] - velocities_circles[j]
                vel_along_normal = relative_velocity.dot(normal)
                overlap = 2 * particle_radius_fraction - distance.norm()
                correction = overlap / 2 * normal
                positions_circles[i] += correction
                positions_circles[j] -= correction
                if vel_along_normal < 0:
                    impulse = -2 * damping_coefficient * vel_along_normal / 2
                    velocities_circles[i] += impulse * normal
                    velocities_circles[j] -= impulse * normal
        for j in ti.static(range(2)):
            if positions_circles[i][j] < (0+particle_radius_fraction) or positions_circles[i][j] > 1 - particle_radius_fraction:
                velocities_circles[i][j] *= -1*damping_coefficient
                positions_circles[i][j] = max(min(positions_circles[i][j], 1 - particle_radius_fraction), 0 + particle_radius_fraction)


@ti.kernel
def bump_circles_with_mouse(mouse_x: ti.f32, mouse_y: ti.f32, mouse_vx: ti.f32, mouse_vy: ti.f32):
    mouse_pos = ti.Vector([mouse_x, mouse_y])
    mouse_velocity = ti.Vector([mouse_vx, mouse_vy])
    for i in range(num_circles):
        distance = positions_circles[i] - mouse_pos
        if distance.norm() < particle_radius_fraction * mouse_size_factor:
            force = -distance.normalized() * mouse_velocity.norm() * 30
            velocities_circles[i] += force

gui = ti.GUI("Particle Physics Simulation", res=(n, n))

initialize_particles()

previous_mouse_x, previous_mouse_y = 0.0, 0.0

while gui.running:
    mouse_x, mouse_y = gui.get_cursor_pos()
    mouse_vx, mouse_vy = (mouse_x - previous_mouse_x, mouse_y - previous_mouse_y)
    bump_circles_with_mouse(mouse_x, mouse_y, mouse_vx, mouse_vy)
    update()
    colors = colors_circles.to_numpy()
    for i, pos in enumerate(positions_circles.to_numpy()):
        if (mouse_x - pos[0])**2 + (mouse_y - pos[1])**2 < (particle_radius_fraction * mouse_size_factor)**2:
            colors[i] = 0xff0000 # Change circle to red if touched by mouse
    
    gui.circles(positions_circles.to_numpy(), radius=particle_radius, color=colors)
    gui.rect((0, 0), (elevator_fraction, 1), color=0xffff00) # Draw elevator area
    gui.show()
    previous_mouse_x, previous_mouse_y = mouse_x, mouse_y