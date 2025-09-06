import taichi as ti

ti.init(arch=ti.gpu)

n = 1024 # Size of canvas in pixels
num_particles = 100 # Number of particles
dt = 0.01 # Time step
particle_radius = 13 # Radius of particles
damping_coefficient = 0.8 # Damping coefficient for collisions between particles and walls
mouse_size_factor = 2 # Size of the mouse interaction area
mouse_speed_factor = 30 # How fast the particles move due to mouse movement
elevator_fraction = 0.2 # Fraction of the screen from the left occupied by the elevator
elevator_acceleration = 2 # Acceleration of the elevator
gravity = -9.8 # Acceleration of gravity

positions_circles = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
velocities_circles = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
accelerations_circles = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
colors_circles = ti.field(dtype=ti.f32, shape=num_particles)

particle_radius_fraction = particle_radius / n
gravity = ti.Vector([0, gravity])

mouse_pos = ti.Vector([0.0, 0.0])
previous_mouse_pos = ti.Vector([0.0, 0.0])
mouse_velocity = ti.Vector([0.0, 0.0])


@ti.kernel
def initialize_particles():
    for i in range(num_particles):
        # Randomize initial positions, velocities, and colors
        positions_circles[i] = ti.Vector([ti.random() * 2, ti.random() * 2])
        velocities_circles[i] = ti.Vector([ti.random() * 2, ti.random() * 2]) * 0.5
        colors_circles[i] = ti.random() * 0xffffff # Random colors

@ti.kernel
def update():
    for i in range(num_particles):
        # Update particle positions based on velocities and accelerations
        if positions_circles[i][0] < elevator_fraction:
            accelerations_circles[i] = ti.Vector([0, elevator_acceleration]) # Particle is in the elevator
        else:
            accelerations_circles[i] = gravity # Particle is outside the elevator
        velocities_circles[i] += accelerations_circles[i] * dt
        positions_circles[i] += velocities_circles[i] * dt

        # Handle collisions between particles
        for j in range(i):
            if i == j:
                continue
            distance = positions_circles[i] - positions_circles[j]
            if distance.norm() < (2 * particle_radius_fraction):
                # Resolve collision
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
                    
        # Handle collisions with walls
        for j in ti.static(range(2)):
            if positions_circles[i][j] < (0+particle_radius_fraction) or positions_circles[i][j] > 1 - particle_radius_fraction:
                velocities_circles[i][j] *= -1*damping_coefficient
                positions_circles[i][j] = max(min(positions_circles[i][j], 1 - particle_radius_fraction), 0 + particle_radius_fraction)


@ti.kernel
def bump_circles_with_mouse(mouse_x: ti.f32, mouse_y: ti.f32, mouse_vx: ti.f32, mouse_vy: ti.f32):
    mouse_pos = ti.Vector([mouse_x, mouse_y])
    mouse_velocity = ti.Vector([mouse_vx, mouse_vy])
    for i in range(num_particles):
        distance = positions_circles[i] - mouse_pos
        if distance.norm() < particle_radius_fraction * mouse_size_factor:
            # Apply force to the particle based on mouse's current velocity
            force = -distance.normalized() * mouse_velocity.norm() * mouse_speed_factor
            velocities_circles[i] += force

gui = ti.GUI("Particle Physics Simulation", res=(n, n))

initialize_particles()

previous_mouse_x, previous_mouse_y = 0.0, 0.0

while gui.running:
    mouse_x, mouse_y = gui.get_cursor_pos()
    mouse_vx, mouse_vy = (mouse_x - previous_mouse_x, mouse_y - previous_mouse_y) # Calculate mouse velocity
    bump_circles_with_mouse(mouse_x, mouse_y, mouse_vx, mouse_vy)
    update()
    
    colors = colors_circles.to_numpy()
    for i, pos in enumerate(positions_circles.to_numpy()):
        if (mouse_x - pos[0])**2 + (mouse_y - pos[1])**2 < (particle_radius_fraction * mouse_size_factor)**2:
            colors[i] = 0xff0000 # Change circle to red if touched by mouse

    gui.circles(positions_circles.to_numpy(), radius=particle_radius, color=colors)
    gui.rect((0, 0), (elevator_fraction, 1), color=0xffff00) # Draw elevator area yellow
    gui.show()
    previous_mouse_x, previous_mouse_y = mouse_x, mouse_y # Update previous mouse position (used for velocity calculation)