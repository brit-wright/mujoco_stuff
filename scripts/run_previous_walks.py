import numpy as np
import mujoco
import glfw
import time

# ===== Load saved states =====
saved_states = np.load('simulation_states.npz', allow_pickle=True)['states']

# ===== Load the MuJoCo model (must match original sim) =====
model_path = './models/go2/maze_with_geoms.xml'  # change to your model file
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


"""
import numpy as np
import mujoco
import glfw
import time

# ===== Load saved states =====
saved_states = np.load('simulation_states.npz', allow_pickle=True)['states']

# ===== Load the MuJoCo model (must match original sim) =====
model_path = './models/go2/maze_with_geoms.xml'  # change to your model file
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# ===== Initialize GLFW =====
if not glfw.init():
    raise Exception("Could not initialize GLFW")
window = glfw.create_window(1920, 1080, "Replay", None, None)
glfw.make_context_current(window)




# ===== Camera & scene setup =====
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scene = mujoco.MjvScene(model, maxgeom=1000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_200)

cam.distance = 8.0
cam.elevation = -35
cam.azimuth = 270

# ===== Replay loop =====
speed = 1.0       # >1 = faster, <1 = slower
skip_frames = 1   # >1 to skip frames for fast-forward

for state in saved_states[::skip_frames]:
    if glfw.window_should_close(window):
        break
    
    nq = model.nq
    nv = model.nv

    print(nq)
    print(nv)

    qpos = state[:nq]
    qvel = state[:nq:nq+nv]

    # Set the saved positions & velocities
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    
    mujoco.mj_forward(model, data)
    cam.lookat[:] = data.qpos[0:3]

    width, height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, width, height)

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
"""