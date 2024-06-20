import pickle
import matplotlib.pyplot as plt

global_accuracy = []
global_f1_SAC = []
global_f1_FedAvg = []
global_f1_Flower = []
local_update = []
reward = []
test_loss = []
reward_FedAvg = []
# Open the .pkl file in read mode
with open("global_f1.pkl", "rb") as file:
    # Load the contents of the .pkl file
    global_f1_SAC = pickle.load(file)
with open("reward.pkl", "rb") as file:
    # Load the contents of the .pkl file
    reward = pickle.load(file)

with open("test_loss.pkl", "rb") as file:
    # Load the contents of the .pkl file
    test_loss = pickle.load(file)


with open(
    "/Users/robert/dev/AINI/HierFL/runs/FedAvg/local-update-15_edgeagg-2_non-iid_alpha-0.01_Jun19_21-43-43/global_f1.pkl",
    "rb",
) as file:
    # Load the contents of the .pkl file
    global_f1_FedAvg = pickle.load(file)

with open(
    "/Users/robert/dev/AINI/HierFL/runs/FedAvg/local-update-15_edgeagg-2_non-iid_alpha-0.01_Jun19_21-43-43/reward.pkl",
    "rb",
) as file:
    # Load the contents of the .pkl file
    reward_FedAvg = pickle.load(file)

with open("/Users/robert/dev/AINI/HierFL/runs/Flower_global_f1.pkl", "rb") as file:
    # Load the contents of the .pkl file
    global_f1_Flower = pickle.load(file)

global_f1_Flower = global_f1_Flower[: len(global_f1_SAC)]
global_f1_FedAvg = global_f1_FedAvg[: len(global_f1_SAC)]

# make the global_f1 smoother
window_size = 10
global_f1_SAC = [
    sum(global_f1_SAC[i : i + window_size]) / window_size
    for i in range(len(global_f1_SAC) - window_size)
]

global_f1_FedAvg = [
    sum(global_f1_FedAvg[i : i + window_size]) / window_size
    for i in range(len(global_f1_FedAvg) - window_size)
]


global_f1_Flower = [
    sum(global_f1_Flower[i : i + window_size]) / window_size
    for i in range(len(global_f1_Flower) - window_size)
]


# Plot global_f1_sac and global_f1_fedavg in the same figure
plt.figure()
plt.plot(global_f1_FedAvg, label="Fixed")
plt.plot(global_f1_SAC, label="SAC")
plt.plot(global_f1_Flower, label="Flower")
plt.xlabel("Communication round")
plt.ylabel("F1 Score")
plt.title("Global F1 of SAC, Fixed and Flower Diagram")
plt.legend()
plt.savefig("f1_diagram.png")

# Plot reward_sac and reward_fedavg in the same figure
# for i in range(len(reward)):
#     reward[i] = reward[i] + 0.0003

# Compare average of reward and reward_FedAvg
# avg_reward = sum(reward) / len(reward)
# avg_reward_FedAvg = sum(reward_FedAvg) / len(reward_FedAvg)
# differece = avg_reward - avg_reward_FedAvg
# print(differece / abs(avg_reward_FedAvg) * 100)
# make the reward smoother
reward_FedAvg = reward_FedAvg[: len(reward)]
window_size = 10
reward = [
    sum(reward[i : i + window_size]) / window_size
    for i in range(len(reward) - window_size)
]
reward_FedAvg = [
    sum(reward_FedAvg[i : i + window_size]) / window_size
    for i in range(len(reward_FedAvg) - window_size)
]


plt.figure()
plt.plot(reward, label="SAC")
plt.plot(reward_FedAvg, label="Fixed")
plt.xlabel("Communication round")
plt.ylabel("Reward")
plt.title("Reward of SAC and Fixed Diagram")
plt.legend()
plt.savefig("reward_diagram.png")
