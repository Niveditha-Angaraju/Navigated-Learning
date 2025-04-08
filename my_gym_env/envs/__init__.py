from .grid_env import GridEnv

from gym.envs.registration import register

register(
    id="GridEnv-v0",
    entry_point="my_gym_env.envs.grid_env:GridEnv",  # Update this if your module path is different
    kwargs={"json_path": "/home/niveditha/Desktop/clg_work/SEM6/extracted_data.json"}  # Updated to JSON
)

