import pickle

file_paths = [
    "global_accuracy.pkl",
    "aggregated_accuracy.pkl",
    "local_update.pkl",
    "reward.pkl",
    "test_loss.pkl",
]
# Open the .pkl file in read mode
for i in file_paths:
    with open(i, "rb") as file:
        # Load the contents of the .pkl file
        data = pickle.load(file)
        print(i[:-4])
        print(data)
