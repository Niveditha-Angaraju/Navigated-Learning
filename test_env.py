import gym
import pickle
from my_gym_env.envs.grid_env import GridEnv

env = GridEnv(render_mode="human", json_path="/home/niveditha/Desktop/clg_work/SEM6/extracted_data.json")
env.load_q_table()  # Load trained Q-table

obs, _ = env.reset()
done = False

while not done:
    action = int(env.q_table[obs].argmax())  # Use learned policy
    obs, reward, done, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}")

    env.render()

env.close()
