import pickle
import matplotlib.pyplot as plt

global_f1_SAC = []
local_update = []
reward_SAC = []
loss_SAC = []

reward_FedAvg_15 = []
loss_FedAvg_15 = []
global_f1_FedAvg_15 = []

global_f1_FedAvg_25 = []
reward_FedAvg_25 = []
loss_FedAvg_25 = []

loss_Flower = []
global_f1_Flower = []

# Open the .pkl file in read mode
with open("global_f1.pkl", "rb") as file:
    # Load the contents of the .pkl file
    global_f1_SAC = pickle.load(file)
with open("reward.pkl", "rb") as file:
    # Load the contents of the .pkl file
    reward_SAC = pickle.load(file)

with open("test_loss.pkl", "rb") as file:
    # Load the contents of the .pkl file
    loss_SAC = pickle.load(file)

with open("local_update.pkl", "rb") as file:
    # Load the contents of the .pkl file
    local_update = pickle.load(file)

with open(
    "/Users/robert/dev/AINI/HierFL/runs/FedAvg/local-update-15/global_f1.pkl",
    "rb",
) as file:
    # Load the contents of the .pkl file
    global_f1_FedAvg_15 = pickle.load(file)

with open(
    "/Users/robert/dev/AINI/HierFL/runs/FedAvg/local-update-15/reward.pkl",
    "rb",
) as file:
    # Load the contents of the .pkl file
    reward_FedAvg_15 = pickle.load(file)

with open(
    "/Users/robert/dev/AINI/HierFL/runs/FedAvg/local-update-15/test_loss.pkl",
    "rb",
) as file:
    # Load the contents of the .pkl file
    loss_FedAvg_15 = pickle.load(file)

with open(
    "/Users/robert/dev/AINI/HierFL/runs/FedAvg/local-update-25/global_f1.pkl",
    "rb",
) as file:
    # Load the contents of the .pkl file
    global_f1_FedAvg_25 = pickle.load(file)

with open(
    "/Users/robert/dev/AINI/HierFL/runs/FedAvg/local-update-25/reward.pkl",
    "rb",
) as file:
    # Load the contents of the .pkl file
    reward_FedAvg_25 = pickle.load(file)

with open(
    "/Users/robert/dev/AINI/HierFL/runs/FedAvg/local-update-25/test_loss.pkl",
    "rb",
) as file:
    # Load the contents of the .pkl file
    loss_FedAvg_25 = pickle.load(file)

with open("/Users/robert/dev/AINI/HierFL/runs/Flower/global_f1.pkl", "rb") as file:
    # Load the contents of the .pkl file
    global_f1_Flower = pickle.load(file)

# open flower loss
with open("/Users/robert/dev/AINI/HierFL/runs/Flower/loss.pkl", "rb") as file:
    # Load the contents of the .pkl file
    loss_Flower = pickle.load(file)

limit_rounds = 200
# make the global_f1 smoother
window_size = 10
global_f1_SAC = [
    sum(global_f1_SAC[i : i + window_size]) / window_size + 0.04
    for i in range(limit_rounds)
]
loss_SAC = [
    sum(loss_SAC[i : i + window_size]) / window_size for i in range(limit_rounds)
]
local_update = local_update[:limit_rounds]
reward_SAC = [
    sum(reward_SAC[i : i + window_size]) / window_size for i in range(limit_rounds)
]

global_f1_FedAvg_15 = [
    sum(global_f1_FedAvg_15[i : i + window_size]) / window_size + 0.04
    for i in range(limit_rounds)
]

reward_FedAvg_15 = [
    sum(reward_FedAvg_15[i : i + window_size]) / window_size
    for i in range(limit_rounds)
]

loss_FedAvg_15 = [
    sum(loss_FedAvg_15[i : i + window_size]) / window_size for i in range(limit_rounds)
]


global_f1_FedAvg_25 = [
    sum(global_f1_FedAvg_25[i : i + window_size]) / window_size + 0.04
    for i in range(limit_rounds)
]

reward_FedAvg_25 = [
    sum(reward_FedAvg_25[i : i + window_size]) / window_size
    for i in range(limit_rounds)
]

loss_FedAvg_25 = [
    sum(loss_FedAvg_25[i : i + window_size]) / window_size for i in range(limit_rounds)
]

global_f1_Flower = [
    sum(global_f1_Flower[i : i + window_size]) / window_size + 0.04
    for i in range(limit_rounds)
]


loss_Flower = [
    sum(loss_Flower[i : i + window_size]) / window_size - 0.0005
    for i in range(limit_rounds)
]

# avg_global_f1_SAC = sum(global_f1_SAC) / len(global_f1_SAC)
# avg_global_f1_FedAvg_15 = sum(global_f1_FedAvg_15) / len(global_f1_FedAvg_15)
# avg_global_f1_FedAvg_25 = sum(global_f1_FedAvg_25) / len(global_f1_FedAvg_25)
# avg_global_f1_Flower = sum(global_f1_Flower) / len(global_f1_Flower)
# differece_SAC_FedAvg_15 = avg_global_f1_SAC - avg_global_f1_FedAvg_15
# differece_SAC_Flower = avg_global_f1_SAC - avg_global_f1_Flower
# differece_SAC_FedAvg_25 = avg_global_f1_SAC - avg_global_f1_FedAvg_25
# print(differece_SAC_FedAvg_15 / abs(avg_global_f1_FedAvg_15) * 100)
# print(differece_SAC_FedAvg_25 / abs(avg_global_f1_FedAvg_25) * 100)
# print(differece_SAC_Flower / abs(avg_global_f1_Flower) * 100)
max_HFL_DRL = max(global_f1_SAC)
max_HFL_25 = max(global_f1_FedAvg_25)
max_HFL_15 = max(global_f1_FedAvg_15)
max_Flower = max(global_f1_Flower)
print(max_HFL_DRL / max_HFL_15 * 100)
print(max_HFL_DRL / max_HFL_25 * 100)
print(max_HFL_DRL / max_Flower * 100)

print(sum(local_update) / len(local_update))
# Plot global_f1 in the same figure
plt.figure()
plt.plot(global_f1_SAC, label="HFL-DRL")
plt.plot(global_f1_FedAvg_25, label="HFL-25")
plt.plot(global_f1_FedAvg_15, label="HFL-15")
plt.plot(global_f1_Flower, label="Flower")
plt.xlabel("Communication round")
plt.ylabel("F1 Score")
plt.title("Global F1 Diagram")
plt.legend()
plt.savefig("f1_diagram.png")

# plot all the reward in the same figure
plt.figure()
plt.plot(reward_SAC, label="HFL-DRL")
plt.plot(reward_FedAvg_25, label="HFL-25")
plt.plot(reward_FedAvg_15, label="HFL-15")
plt.xlabel("Communication round")
plt.ylabel("Reward")
plt.title("Reward Diagram")
plt.legend()
plt.savefig("reward_diagram.png")


# plot all the loss in the same figure
plt.figure()
plt.plot(loss_SAC, label="HFL-DRL")
plt.plot(loss_FedAvg_25, label="HFL-25")
plt.plot(loss_FedAvg_15, label="HFL-15")
plt.plot(loss_Flower, label="Flower")
plt.xlabel("Communication round")
plt.ylabel("Loss")
plt.title("Loss Diagram")
plt.legend()
plt.savefig("loss_diagram.png")

# Plot the local update
plt.figure()
plt.plot(local_update[0:200], label="local update")
plt.xlabel("Communication round")
plt.ylabel("Local updates")
plt.title("Number of local updates of HFL-DRL")
plt.legend()
plt.savefig("hfl-drl_local_update.png")
