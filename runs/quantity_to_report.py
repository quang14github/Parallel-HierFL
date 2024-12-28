import os
import pandas as pd
import pickle

report_data = []
baseline = None
# Iterate through directories and files
for root, dirs, files in os.walk("runs/quantity"):
    if (
        "global_accuracy.pkl" in files
        # and "local_update.pkl" in files
        and "training_time.pkl" in files
    ):
        # Read global_accuracy.pkl
        with open(os.path.join(root, "global_accuracy.pkl"), "rb") as f:
            global_accuracy = pickle.load(f)[:300]

        # if "FedAvg" in os.path.join(root, "global_accuracy.pkl"):
        #     baseline = max(global_accuracy)
        #     print(baseline)
        # print(root)

        # # Read local_update.pkl
        # with open(os.path.join(root, "local_update.pkl"), "rb") as f:
        #     local_updates = pickle.load(f)

        # # Read training_time.txt
        # with open(os.path.join(root, "training_time.pkl"), "rb") as f:
        #     training_time = pickle.load(f)

        # Calculate max accuracy and rounds to baseline
        max_accuracy = max(global_accuracy)
        # rounds_to_max_accuracy = next(
        #     (i for i, acc in enumerate(global_accuracy) if acc >= max_accuracy), None
        # )
        # rounds_to_baseline = next(
        #     (i for i, acc in enumerate(global_accuracy) if acc >= baseline), None
        # )

        # training_time_to_baseline = (
        #     training_time[rounds_to_baseline]
        #     if rounds_to_baseline is not None
        #     else None
        # )

        # Calculate total local updates and total training time
        # edgeagg = int(root.split("/")[2].split("_")[1].split("-")[1])
        # total_local_updates_baseline = (
        #     sum(local_updates[:rounds_to_baseline])
        #     if rounds_to_baseline is not None
        #     else None
        # ) * edgeagg
        # total_local_updates = sum(local_updates) * edgeagg
        # total_training_time = (
        #     round(training_time / total_local_updates * total_local_updates_baseline)
        #     if rounds_to_baseline is not None
        #     else None
        # )

        # Append the gathered information to the report data
        report_data.append(
            {
                "Scenario": ("/").join(root.split("/")[1:]),
                "Max Accuracy": max_accuracy,
                # f"Rounds to Max Accuracy": rounds_to_max_accuracy,
                # f"Rounds to Baseline ({baseline})": rounds_to_baseline,
                # # "Total Local Updates": total_local_updates_baseline,
                # "Training Time to Baseline (s)": training_time_to_baseline,
            }
        )

# Create a DataFrame from the report data
report_df = pd.DataFrame(report_data)
print(report_df)
# to excel file
# report_df.to_excel("quantity_report.xlsx", index=False)
