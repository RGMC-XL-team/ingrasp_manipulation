# Robotic In-Hand Manipulation for Large-Range Precise Object Movement: The RGMC Champion Solution

[[Project Website](https://rgmc-xl-team.github.io/ingrasp_manipulation)]

Repository for the paper _Robotic In-Hand Manipulation for Large-Range Precise Object Movement: The RGMC Champion Solution_, submitted to RA-L, 2024.

In this repository, we provide:

- The algorithm for our proposed in-grasp object movement approach.
- A simulator based on MuJoCo for a quick test of the algorithm.

We do not provide code related to the hardware implementation, as it depends on your specific hardware setup.

<div align="center">
  <img src="./docs/ingrasp_manipulation_simulation.gif" alt="MuJoCo simulation" width="50%" />
</div>

## Installation

1. Install ROS Noetic.

1. Create a conda virtual env and install [Pinocchio](https://github.com/stack-of-tasks/pinocchio):

   ```bash
   conda create -n <YOUR_ENV_NAME> -c conda-forge python=3.9 pinocchio
   ```

1. In your python env:

   ```bash
   pip install mujoco==3.1.3
   pip install dm_control==1.0.16
   pip install rospkg
   ```

1. Clone this repo into a ROS workspace.

1. Build the workspace:
   ```
   cd <YOUR_WORKSPACE>
   catkin build
   ```

## Usage (Simulation)

Run the mujoco simulation only (no motion):

```bash
# in your python env
cd leap_task_A/scripts
source ../../../../devel/setup.bash
python leaphand_mujoco.py
```

Run the in-grasp object movement:

```bash
# in your python env
cd leap_task_A/scripts
source ../../../../devel/setup.bash
python leaphand_control.py
```
