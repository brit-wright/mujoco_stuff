# Project Overview
Hello! Thanks for dropping by! This repo stores files for 'phase 3' of research I did for Summer 2025. The 
goal of my project was to develop a path-planning algorithm and model-based controller for the Unitree Go2 
quadruped. [Phase 1](https://github.com/brit-wright/Research-Code) involved implementing CPU-based RRT and GPU-parallelized RRT planners. 
[Phase 2](https://github.com/brit-wright/IRIS_env_drake) involved using convex decomposition through 
[IRIS and pydrake](https://drake.mit.edu/pydrake/) to decompose a map into convex regions of freespace
and using it to plan. Finally, Phase 3 involved building and simulating a model-based controller for the 
Go2 quadruped in Mujoco simulation.

# Software/Library Requirements
This repo is entirely Python-based. The following libraries are needed: numpy, matplotlib, shapely, 
scipy, torch, mujoco-py, glfw, and optionally cProfile (for time-profiling). Since part of this project involves
GPU-parallelization, a CUDA-capable machine is recommended, but if your machine doesn't have a GPU,
setting device='cpu' will work for testing/debugging, just no actual parallelization.

# About the branches
For the most recent (and a better organized) version of this project, head over to the 'branch1' branch. The
main files used for walking are walk_to_multiple_locationsv2.py and QuadrupedControllerv1_2.py

# Questions
Feel free to reach out at bmwright@caltech.edu if you have questions :D
