import numpy as np
import mujoco
import glfw
import time

# ===== Load saved states =====
# saved_states = np.load('simulation_states.npz', allow_pickle=True)['states']
# saved_states = np.load('scripts/recordings/noel_maze_walk_turn_point_8.npz', allow_pickle=True)['states']
saved_states = np.load('/home/brittany/Mujoco_Example/scripts/recordings/simulation_states_8_20_2025_1_19.xml.npz', allow_pickle=True)['states']

# ===== Load the MuJoCo model (must match original sim) =====
model_path = '/home/brittany/Mujoco_Example/models/go2/maze2_with_geoms_8_20_2025_1_19.xml'  # change to your model file
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# ===== Initialize GLFW =====
if not glfw.init():
    raise Exception("Could not initialize GLFW")
window = glfw.create_window(1920, 1080, "Replay", None, None)
glfw.make_context_current(window)

width, height = glfw.get_framebuffer_size(window)
viewport = mujoco.MjrRect(0, 0, width, height)

# ===== Camera & scene setup =====
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_200)

cam.distance = 8.0
cam.elevation = -35
cam.azimuth = 270

# ===== Replay loop =====
speed = 5.0       # >1 = faster, <1 = slower
skip_frames = 10   # >1 to skip frames for fast-forward

for state in saved_states[::skip_frames]:
    if glfw.window_should_close(window):
        break
    
    # Set the saved positions & velocities
    data.qpos[:] = state['qpos']
    data.qvel[:] = state['qvel']

    mujoco.mj_forward(model, data)
    cam.lookat[:] = data.qpos[0:3]
    
    # Render current state
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)
    
    glfw.swap_buffers(window)
    glfw.poll_events()
    
    # Sleep based on recorded timestep & desired speed
    dt = (saved_states[1]['time'] - saved_states[0]['time']) / speed
    time.sleep(dt)
    # print('running')

# ===== Cleanup =====
glfw.destroy_window(window)
glfw.terminate()