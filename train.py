import time
import torch
from torch import nn, optim
from utils import *
import args
from models.transformers import GTN_VAE
import pandas as pd
import os
import seaborn as sns
import random
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(
    model,
    optimizer,
    x,
    pos_enc,
    train_adj,
    val_adj,
    sampled_vae_loss,
    sampled_metrics,
    num_epochs=200,
    train_samples=8000,
    val_samples=400,
    device=None
):
    if device is None:
        # infer device from model parameters
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # ensure inputs are on device (caller may already have moved them)
    x = x.to(device)
    pos_enc = pos_enc.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_roc": [],
        "val_roc": [],
        "train_ap": [],
        "val_ap": [],
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        adj_recon = model(x, pos_enc, train_adj)  # forward on training adjacency

        total_loss, recon_loss, kl_loss = sampled_vae_loss(
            adj_recon, train_adj, model.z_mean, model.z_logvar, num_samples=train_samples, beta=args.KL_BETA
        )

        total_loss.backward()

        optimizer.step()

        # Metrics (sampled) on train
        roc_train, ap_train, acc_train = sampled_metrics(adj_recon, train_adj, num_samples=train_samples)

        model.eval()
        with torch.no_grad():
            adj_val_recon = model(x, pos_enc, val_adj)
            val_loss, _, _ = sampled_vae_loss(
                adj_val_recon, val_adj, model.z_mean, model.z_logvar, num_samples=val_samples, beta=args.KL_BETA
            )
            roc_val, ap_val, acc_val = sampled_metrics(adj_val_recon, val_adj, num_samples=val_samples)

        print(
            f"Epoch [{epoch:03d}/{num_epochs}] | "
            f"Train Loss: {total_loss.item():.4f} | Val Loss: {val_loss.item():.4f} | "
            f"Train Acc: {acc_train:.4f} | Val Acc: {acc_val:.4f} | "
            f"Train ROC: {roc_train:.4f} | Val ROC: {roc_val:.4f} | "
            f"Train AP: {ap_train:.4f} | Val AP: {ap_val:.4f}"
        )

        # record history
        history["train_loss"].append(total_loss.item())
        history["val_loss"].append(val_loss.item())
        history["train_acc"].append(acc_train)
        history["val_acc"].append(acc_val)
        history["train_roc"].append(roc_train)
        history["val_roc"].append(roc_val)
        history["train_ap"].append(ap_train)
        history["val_ap"].append(ap_val)

    return history

def test(
    model,
    x,
    pos_enc,
    test_adj,
    sampled_metrics,
    experiments,
    num_samples=5000,
    device=None,
):
        # torch.manual_seed(42)
    def test_model(model, x, pos_e, test_adj, num_samples):
        model.eval()
        with torch.no_grad():
            adj_test_recon = model(x, pos_enc, test_adj)
        
            # Balanced sampled VAE loss
            test_loss, _, _ = sampled_vae_loss(
                adj_test_recon, test_adj, model.z_mean, model.z_logvar, beta=args.KL_BETA, num_samples=1054

            )
        
            # Sampled metrics (balanced)
            roc_test, ap_test, acc_test = sampled_metrics(
                adj_test_recon, test_adj, num_samples=num_samples
            )
        return test_loss, acc_test, roc_test, ap_test

    exps = experiments
    losses = []
    accs = []
    rocs = []
    precs = []
    for i in range(exps):
        torch.manual_seed(42 + i)
        t_loss, t_acc, t_roc, t_prec = test_model(model, x, pos_enc, test_adj, num_samples=num_samples)
        losses.append(t_loss.detach().cpu().item())
        accs.append(t_acc)
        rocs.append(t_roc)
        precs.append(t_prec)
    losses = np.array(losses)
    accs = np.array(accs)
    rocs = np.array(rocs)
    precs = np.array(precs)
    print("\n=== FINAL TEST RESULTS ===")
    print(
        f"Test Loss: {losses.mean():.4f} +- {losses.std()} | \n"
        f"Test Acc: {accs.mean():.4f} +- {accs.std()} | \n"
        f"ROC-AUC: {rocs.mean():.4f} +- {rocs.std()} |\nAP: {precs.mean():.4f} +- {precs.std()}"
    )


def _plot_melted(df, cols, title, fname):
            plt.figure(figsize=(8, 5))
            data = df.melt(id_vars="epoch", value_vars=cols, var_name="variable", value_name="value")
            sns.lineplot(x="epoch", y="value", hue="variable", data=data)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(args.VIS_PATH, fname))
            plt.close()

def visualize(history):
        os.makedirs(args.VIS_PATH, exist_ok=True)
        # convert history dict to DataFrame and add epoch column
        df = pd.DataFrame(history)
        df["epoch"] = range(1, len(df) + 1)

        sns.set_theme(style="whitegrid")
        # helper to plot and save

        _plot_melted(df, ["train_loss", "val_loss"], "Training / Validation Loss", "loss.png")
        _plot_melted(df, ["train_acc", "val_acc"], "Training / Validation Accuracy", "accuracy.png")
        _plot_melted(df, ["train_roc", "val_roc"], "Training / Validation ROC AUC", "roc.png")
        _plot_melted(df, ["train_ap", "val_ap"], "Training / Validation Average Precision", "ap.png")

        # save raw history for later analysis
        df.to_csv(os.path.join(args.VIS_PATH, f"history{time.time()}.csv"), index=False)

if __name__ == "__main__":
    set_seed(42)
    x, pos_enc, train_adj, val_adj, test_adj, y = prepare_gtnvae_data(args.DATA_DIR, args.DATASET, pos_dim=args.POS_DIM)
    device = args.DEVICE

    print(
        f"\n===== DATASET SUMMARY =====\n"
        f"Node Feature Shape (x):           {x.shape}  -> [B, N, d_node_in]\n"
        f"Positional Encoding Shape:        {pos_enc.shape}  -> [B, N, d_pos]\n"
        f"Train Adjacency Shape:            {train_adj.shape}  -> [B, N, N]\n"
        f"Validation Adjacency Shape:       {val_adj.shape}  -> [B, N, N]\n"
        f"Test Adjacency Shape:             {test_adj.shape}  -> [B, N, N]\n"
        f"--------------------------------------\n"
        f"Total Nodes (N):                  {train_adj.size(-1)}\n"
        f"Feature Dimension (d_node_in):    {x.size(-1)}\n"
        f"Positional Dim (d_pos):           {pos_enc.size(-1)}\n"
        f"Total Train Edges:                {int(train_adj.sum().item())}\n"
        f"Total Val Edges:                  {int(val_adj.sum().item())}\n"
        f"Total Test Edges:                 {int(test_adj.sum().item())}\n"
        f"======================================\n"
    )

    if args.LOAD:
        print(f"Loading model from {args.MODEL_PATH}...")
        model = torch.load(args.MODEL_PATH, weights_only=False).to(device)
    else:
        model = GTN_VAE(
            input_dim=x.size(-1),
            pos_dim=pos_enc.size(-1),
            hidden_dim=args.HIDDEN_DIM,
            n_layers=args.NUM_LAYERS,
            n_heads=args.NUM_HEADS
        ).to(device)
    x, pos_enc, train_adj, val_adj, test_adj, y = (
        x.to(device),
        pos_enc.to(device),
        train_adj.to(device),
        val_adj.to(device),
        test_adj.to(device),
        y.to(device)
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.LR, weight_decay=5e-4)
    history = train(
        model,
        optimizer,
        x,
        pos_enc,
        train_adj,
        val_adj,
        sampled_vae_loss=sampled_vae_loss,
        sampled_metrics=sampled_metrics,
        num_epochs=args.EPOCHS,
        train_samples=args.TRAIN_SAMPLES,
        val_samples=args.VAL_SAMPLES,
        device=device
    )
    if args.SAVE:
        torch.save(model, args.MODEL_PATH)
        print(f"Model saved to {args.MODEL_PATH}")

    if args.TRAIN_VISUALIZATIONS:
        visualize(history)
    
    # Evaluate on test set
    test(
        model.eval(),
        x.to(device),
        pos_enc.to(device),
        test_adj.to(device),
        sampled_metrics=sampled_metrics,
        num_samples=args.TEST_SAMPLES,
        device=device,
        experiments=args.EXPERIMENTS
    )
    print("Testing complete.")



