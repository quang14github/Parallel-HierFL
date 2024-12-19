import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

idx = 0
# Path to the runs folder
runs_folder = "/Users/robert/dev/AINI/HierFL/runs/"
# Iterate through each file in the runs folder
for folder in os.listdir(runs_folder):
    folder_path = os.path.join(runs_folder, folder)
    if os.path.isdir(folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".pkl"):
                    file_path = os.path.join(root, file)
                    # Read the data file into DataFrame
                    try:
                        with open(file_path, "rb") as f:
                            data = pickle.load(f)
                        if file == "global_accuracy.pkl":
                            # print the first index when global_accuracy is greater or equal to 0.935
                            # for index, value in enumerate(data):
                            #     if value >= 0.935:
                            #         idx = index
                            #         break
                            print(max(data))
                    except Exception as e:
                        print(f"Could not read {file_path}: {e}")
                        continue

                    # if file == "local_update.pkl":
                    # print the average of local update
                    # print(data.mean())
                    # Plot the data using Seaborn
                    sns.set(style="darkgrid")
                    plt.figure(figsize=(10, 6))
                    sns.lineplot(data=data)
                    plt.title(f"Plot for {file}")
                    # Plot a baseline of maximum global_accuracy

                    # Determine the result folder
                    result_folder = os.path.join(root, "result")

                    # Create the result folder if it doesn't exist
                    os.makedirs(result_folder, exist_ok=True)

                    # Save the figure as a PNG file in the result folder
                    output_file_path = os.path.join(
                        result_folder, f"{os.path.splitext(file)[0]}.png"
                    )
                    plt.savefig(output_file_path)
                    plt.close()  # Close the figure to free memory
# print(f"rounds: {idx+1}")
# with open(os.path.join(runs_folder, "local_update.pkl"), "rb") as f:
#     data = pickle.load(f)
#     print(f"total local updates: {sum(data[:idx])}")

# # read training time from txt fiel
# with open(os.path.join(runs_folder, "training_time.txt"), "r") as f:
#     x = float(f.read())
#     print(f"training time: {round(x/sum(data)*sum(data[:idx]))}s")
