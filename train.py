from my_gym_env.envs.grid_env import GridEnv

# Initialize environment
env = GridEnv(json_path="/home/niveditha/Desktop/clg_work/SEM6/extracted_data.json")

# Train Q-learning agent
print("Training Q-learning agent...")
env.train_q_learning(episodes=5000)  # Train for 5000 episodes

print("Training completed. Q-table saved.")
