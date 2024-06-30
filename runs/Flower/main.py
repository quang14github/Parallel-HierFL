import pickle
import matplotlib.pyplot as plt

global_acc = []
global_f1 = []
loss = []

with open("Flower_global_acc.pkl", "rb") as f:
    global_acc = pickle.load(f)

with open("Flower_global_f1.pkl", "rb") as f:
    global_f1 = pickle.load(f)

with open("Flower_loss.pkl", "rb") as f:
    loss = pickle.load(f)

# limit the number of rounds to 500
global_acc = global_acc[:250]
global_f1 = global_f1[:250]
loss = loss[:250]

# Smooth the global accuracy and f1 score
window_size = 10
global_acc = [
    sum(global_acc[i : i + window_size]) / window_size
    for i in range(len(global_acc) - window_size)
]
global_f1 = [
    sum(global_f1[i : i + window_size]) / window_size
    for i in range(len(global_f1) - window_size)
]

loss = [
    sum(loss[i : i + window_size]) / window_size for i in range(len(loss) - window_size)
]

# plot the global accuracy
plt.figure()
plt.plot(global_acc, label="global accuracy")
plt.xlabel("Communication round")
plt.ylabel("Accuracy")
plt.title("Global Accuracy of Flower")
plt.legend()
plt.savefig("Flower_global_acc.png")

# plot the global f1
plt.figure()
plt.plot(global_f1, label="global f1")
plt.xlabel("Communication round")
plt.ylabel("F1 Score")
plt.title("Global F1 of Flower")
plt.legend()
plt.savefig("Flower_global_f1.png")

# plot the loss
plt.figure()
plt.plot(loss, label="loss")
plt.xlabel("Communication round")
plt.ylabel("Loss")
plt.title("Loss of Flower")
plt.legend()
plt.savefig("Flower_loss.png")
