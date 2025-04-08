import pickle
import numpy as np

with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

print(q_table)
