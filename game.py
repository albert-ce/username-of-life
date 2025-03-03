import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, HPacker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation


TEXT = "/conway"    # Replace by your GitHub username
FONT_SIZE = 50
FONT_NAME = "Segoe UI"
WIDTH = 600
HEIGHT = 100
DPI = 40
SPACE_GAP = 20

N_ITERATIONS = 200
GIF_NAME = "username_of_life.gif"
INITIAL_DELAY = 15
SCALE_FACTOR = 6
INIT_STATE_ALPHA = 0


def text2bool(text=TEXT, font_size=FONT_SIZE, font_name=FONT_NAME, width=WIDTH, height=HEIGHT, dpi=DPI, gap=SPACE_GAP):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=dpi)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    icon = plt.imread("github_icon.png")
    original_height = icon.shape[0]
    zoom = 0.5 * height / original_height
    imagebox = OffsetImage(icon, zoom=zoom)
    
    texto = TextArea(text, textprops=dict(color="black", size=font_size, fontname=font_name))
    packed = HPacker(children=[imagebox, texto], align="center", pad=font_size/2, sep=gap)
    ab = AnnotationBbox(packed, (0, 0.5), box_alignment=(0,0.5), xycoords='axes fraction', frameon=False)
    ax.add_artist(ab)
    ax.axis("off")
    fig.canvas.draw()
    text_image = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    bool_img = np.all(text_image[:, :, :3] < 160, axis=-1)
    return bool_img

def game_of_life_step(grid):
    neighbors = sum(np.roll(np.roll(grid, i, 0), j, 1)
                 for i in (-1, 0, 1) for j in (-1, 0, 1)
                 if (i != 0 or j != 0))
    return (neighbors == 3) | (grid & (neighbors == 2))

def scale_array(array, scale_factor, grid_color=0):
    height, width = array.shape
    new_height = height * (scale_factor + 1) - 1
    new_width = width * (scale_factor + 1) - 1
    scaled_array = np.full((new_height, new_width), grid_color, dtype=array.dtype)
    
    for i in range(height):
        for j in range(width):
            scaled_array[i*(scale_factor+1):i*(scale_factor+1)+scale_factor, 
                         j*(scale_factor+1):j*(scale_factor+1)+scale_factor] = array[i, j]
    
    return scaled_array

def make_gif(bool_img, n_iterations=N_ITERATIONS, gif_name=GIF_NAME, initial_delay=INITIAL_DELAY, scale_factor=SCALE_FACTOR, alpha=INIT_STATE_ALPHA):
    grid = bool_img.copy()
    grid_scaled = scale_array(grid, scale_factor)
    
    initial_state_scaled = grid_scaled.copy()
    
    fig, ax = plt.subplots(figsize=(grid_scaled.shape[1]/100, grid_scaled.shape[0]/100), dpi=100)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis("off")
    
    colors = [(13/255, 17/255, 23/255), (240/255, 246/255, 252/255)]
    cmap_custom = LinearSegmentedColormap.from_list("custom_gray", colors)
    
    im = ax.imshow(grid_scaled, cmap=cmap_custom, interpolation='nearest')
    initial_state_layer = ax.imshow(initial_state_scaled, cmap=cmap_custom, alpha=alpha, interpolation='nearest')

    def update(frame):
        nonlocal grid, grid_scaled
        if frame >= initial_delay:
            grid = game_of_life_step(grid)
            grid_scaled = scale_array(grid, scale_factor)
            im.set_data(grid_scaled)
        return [im, initial_state_layer]

    total_frames = initial_delay + n_iterations

    ani = FuncAnimation(fig, update, frames=total_frames, blit=True)
    ani.save(gif_name, writer='pillow', fps=10)

if __name__ == "__main__":
    print("Rendering the initial state...")
    bool_img = text2bool()
    print("Initial state successfully calculated ✔")
    print("Rendering the Game of Life animation...")
    make_gif(bool_img)
    print("Animation completed and saved ✔")
