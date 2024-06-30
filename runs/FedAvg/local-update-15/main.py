import pickle

file_paths = [
    "global_accuracy.pkl",
    "global_f1.pkl",
    "local_update.pkl",
    "reward.pkl",
    "test_loss.pkl",
]
# Open the .pkl file in read mode
for i in file_paths:
    with open(i, "rb") as file:
        # Load the contents of the .pkl file
        data = pickle.load(file)
        print(len(data))
