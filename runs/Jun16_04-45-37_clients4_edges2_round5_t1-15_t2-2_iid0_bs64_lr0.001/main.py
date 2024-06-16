import pickle

# Specify the path to your .pkl file
file_path = "global_accuracy.pkl"

# Open the .pkl file in read mode
with open(file_path, "rb") as file:
    # Load the contents of the .pkl file
    data = pickle.load(file)

# Now you can work with the loaded data
# For example, you can print it
print(data)
