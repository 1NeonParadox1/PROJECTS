import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


g = 9.81
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

velocity = 20
angle = 45


def projectile(angle, velocity):
   

    angle_rad = np.radians(angle)

    vx = velocity * np.cos(angle_rad)
    vy = velocity * np.sin(angle_rad)

    t_flight = (2 * vy) / g

    t = np.linspace(0, t_flight, 100)   # 0~t_flight ko 100 equal bits me seprate kr diya hai

    x = vx * t
    y = vy * t - 0.5 * g * t**2

    range = (velocity**2 * np.sin(2*angle_rad))/g
    max_height = (velocity**2 * np.sin(angle_rad)**2)/(2*g)
    x_max = vx * (vy/g)
    return x, y, range, max_height, t_flight, x_max


x, y, range, max_height, t_flight, x_max = projectile(angle, velocity)
line, = plt.plot(x, y)
max_point, = ax.plot(x_max, max_height, 'ro')

range_text = ax.text(0.02, 0.95, f"Range = {range:.2f} m",
                     transform=ax.transAxes)

height_text = ax.text(0.02, 0.90, f"Max Height = {max_height:.2f} m",
                      transform=ax.transAxes)

print("Range of Projectile: ", range)
print("Maximum Height attained: ", max_height, "At time: ", t_flight/2)

plt.xlabel("Horizontal Distance (m)")
plt.ylabel("Vertical Height (m)")
plt.title("Projectile Motion Trajectory")


ax_angle = plt.axes([0.2, 0.1, 0.65, 0.03])
angle_slider = Slider(ax_angle, 'Angle', 0, 90, valinit=45)

ax_velocity = plt.axes([0.2, 0.05, 0.65, 0.03])
velocity_slider = Slider(ax_velocity, 'Velocity', 1, 50, valinit=20)

def update(val):

    angle = angle_slider.val
    velocity = velocity_slider.val

    x, y, range_proj, max_height, t_flight, x_max = projectile(angle, velocity)

    line.set_xdata(x)
    line.set_ydata(y)

    max_point.set_xdata([x_max])
    max_point.set_ydata([max_height])

    range_text.set_text(f"Range = {range_proj:.2f} m")
    height_text.set_text(f"Max Height = {max_height:.2f} m")

    ax.relim()
    ax.autoscale_view()

    fig.canvas.draw_idle()
angle_slider.on_changed(update)
velocity_slider.on_changed(update)
plt.show()