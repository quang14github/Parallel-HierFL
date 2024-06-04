import matplotlib.pyplot as plt
import numpy as np

expected_maximum = 0.9924
# Read data from the text file
hierfl_data = []
with open("evaluation_results.txt", "r") as file:
    for line in file:
        round, accuracy = line.split()
        hierfl_data.append((int(round), float(accuracy)))
# Separate the round and accuracy values into separate lists
rounds = [d[0] for d in hierfl_data]
accuracies = [d[1] for d in hierfl_data]

flower_accuracies = []
with open("flower_evaluation.txt", "r") as file:
    for line in file:
        flower_accuracies.append(float(line))

fig = plt.figure()
axis = fig.add_subplot(111)

plt.axhline(
    y=expected_maximum,
    color="r",
    linestyle="--",
    label=f"Paper's best result @{expected_maximum}",
)
# plot the data with round on the x-axis and accuracy on the y-axis
plt.plot(rounds, accuracies, label="HIER-FL")
plt.plot(range(0, len(flower_accuracies)), flower_accuracies, label="FLower")
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title("HIER-FL vs FLower")
plt.legend(loc="lower right")
xleft, xright = axis.get_xlim()
ybottom, ytop = axis.get_ylim()
axis.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * 1.0)
plt.savefig("hierfl_vs_flower.png")
plt.show()
