import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from tqdm import tqdm

saving_dir = "figs"

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs(f"davebulaval/KILL")

model_name_to_plot = ["camembertv2-base"]

da_eval_loss = []
no_da_eval_loss = []
da_train_loss = []
no_da_train_loss = []
for run in tqdm(runs):
    if run.state == "finished":
        model_name = run.metadata.get("args")[3].split("=")[1].split("/")[-1]

        if model_name in model_name_to_plot:
            with_da = run.metadata.get("args")[1].split("=")[-1]
            eval_data = [
                x.get("eval/loss")
                for x in run.scan_history(keys=["eval/loss"], page_size=2000)
            ]
            train_data = [
                x.get("train/loss")
                for x in run.scan_history(keys=["train/loss"], page_size=2000)
            ]
            if with_da == "true":
                da_eval_loss.append(eval_data)
                da_train_loss.append(train_data)
            else:
                no_da_eval_loss.append(eval_data)
                no_da_train_loss.append(train_data)

num_epoch = 100
epoch = np.array([[i] * 10 for i in range(0, num_epoch)] * 2).flatten()
run_name = list(range(0, 10)) * num_epoch * 2

hue = ["Train"] * 1000 + ["Eval"] * 1000
data_da = pd.DataFrame(
    {
        "Epoch": epoch,
        "run": run_name,
        "Loss": list(np.array(da_train_loss).transpose().flatten())
        + list(np.array(da_eval_loss).transpose().flatten()),
        "Metric": hue,
    }
)

hue = ["Train"] * 1000 + ["Eval"] * 1000
data_no_da = pd.DataFrame(
    {
        "Epoch": epoch,
        "run": run_name,
        "Loss": list(np.array(no_da_train_loss).transpose().flatten())
        + list(np.array(no_da_eval_loss).transpose().flatten()),
        "Metric": hue,
    }
)

ax = sns.relplot(
    x="Epoch",
    y="Loss",
    hue="Metric",
    data=data_da,
    kind="line",
    errorbar="sd",
    palette="viridis",
)
ax.set(ylim=(0, 40))
leg = ax._legend
leg.set_bbox_to_anchor([0.95, 0.7])
leg.set_title("")
plt.tight_layout()

plt.savefig("da_loss_fig.png", dpi=300)
plt.show()

ax = sns.relplot(
    x="Epoch",
    y="Loss",
    hue="Metric",
    data=data_no_da,
    kind="line",
    errorbar="sd",
    palette="magma",
)
ax.set(ylim=(0, 40))
leg = ax._legend
leg.set_bbox_to_anchor([0.95, 0.7])
leg.set_title("")

plt.tight_layout()

plt.savefig("no_da_loss_fig.png", dpi=300)
plt.show()
