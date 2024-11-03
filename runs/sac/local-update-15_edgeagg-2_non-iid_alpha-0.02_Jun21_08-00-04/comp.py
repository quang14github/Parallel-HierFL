import pickle
import matplotlib.pyplot as plt

local_update = []
num_round = 200

with open("local_update.pkl", "rb") as file:
    # Load the contents of the .pkl file
    local_update = pickle.load(file)

total_SAC = sum(local_update[:num_round]) * 2
total_HFL_25 = 25 * 2 * num_round
total_HFL_15 = 15 * 2 * num_round


total_flower = 11838 / 64 * num_round / 4

# Plotting the stack bar diagram
labels = ["HFL-DRL", "HFL-25", "HFL-15", "Flower"]
totals = [total_SAC, total_HFL_25, total_HFL_15, total_flower]
for i in totals:
    print(i)
plt.bar(labels, totals)
plt.xlabel("Network traffic classification methods")
plt.ylabel("Number of local updates")
plt.title("Total number of local updates for each methods")

# save the plot
plt.savefig("local_update_comp.png")
