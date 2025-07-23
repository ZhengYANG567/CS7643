from tqdm import tqdm
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import os
import time
from util import *

class CNN_FC(nn.Module):
    def __init__(self):
        super(CNN_FC, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input dimension [3, 512, 768]

        # self.model = models.resnet50(weights=None)
        # self.model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        # self.model = models.vit_b_32(weights='IMAGENET1K_V1')

        # self.fc = nn.Sequential(
            # for resnet50
        #     nn.Linear(1000, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 2)
        # )
        self.model = nn.Sequential(
            nn.Conv2d(24, 256, 7, stride=1, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 5, stride=1, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 5, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, stride=1, padding="same"),
        )

    def forward(self, x):
        out = self.model(x)
        return out
        # out = out.view(out.size()[0], -1)
        # return self.fc(out)

def train_model(model, optimizer, train_loader, valid_loader,
    n_epochs=100, patience=25, criterion=nn.CrossEntropyLoss(),
    train_losses=[], valid_losses=[], prefix="CNN"):

    # Initialize trackers, these are not parameters and should not be changed
    stale = 0
    best_MSE = 1e+9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _train():
        model.train()
        train_loss = []

        for batch in tqdm(train_loader):
            imgs, labels = batch
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            # acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            train_loss.append(loss.item())
            # train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        # train_acc = sum(train_accs) / len(train_accs)
        train_losses.append(train_loss)
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5g}")

    def _validate(record=True):
        model.eval()
        valid_loss = []
        # valid_accs = []
        nonlocal best_MSE

        for batch in tqdm(valid_loader):
            imgs, labels = batch
            with torch.no_grad():
                logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            # acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            valid_loss.append(loss.item())
            # valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        # valid_acc = sum(valid_accs) / len(valid_accs)
        if record:
            valid_losses.append(valid_loss)
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5g}")


        # update logs
        if valid_loss < best_MSE:
            with open(f"./{prefix}_log.txt","a"):
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5g} -> best")
        else:
            with open(f"./{prefix}_log.txt","a"):
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5g}")
        return valid_loss

    if len(train_losses) > 0:
        epoch = 0
        best_MSE = _validate(record=False)

    for epoch in range(len(train_losses), n_epochs):

        start_time = time.time()
        # ---------- Training ----------
        _train()

        # ---------- Validation ----------
        valid_loss = _validate()

        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch {epoch+1} duration: {epoch_duration:.2f} seconds")
        # save models
        if valid_loss < best_MSE:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(), f"{prefix}_best.ckpt") # only save best
            with open(f"{prefix}_best.sum", "w") as fsum:
                fsum.write(generate_checksum(f"{prefix}_best.ckpt"))
            best_MSE = valid_loss
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break
        with open(f"{prefix}_loss.pkl", "wb") as fout:
            pickle.dump({"train": train_losses, "validation": valid_losses}, fout)
    return train_losses, valid_losses

def load_model(model_path, loss_path = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_FC()
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    if loss_path != None:
        with open(loss_path, "rb") as fin:
            losses = pickle.load(fin)
        return model, losses["train"], losses["validation"]
    else:
        return model

def try_load_model(exp_name):
    model_name = f"{exp_name}_best.ckpt"
    loss_name = f"{exp_name}_loss.pkl"
    cksum_name = f"{exp_name}_best.sum"
    if os.path.exists(model_name):
        if os.path.exists(cksum_name):
            with open(cksum_name, "r") as fsum:
                cksum_in_file = fsum.read()
            cksum_loaded = generate_checksum(model_name)
            if cksum_loaded != cksum_in_file:
                logging.warning("Check sum does not match! Is your model checkpoint correct?")
        else:
            logging.warning("Check sum file not found")
        if os.path.exists(loss_name):
            model, train_loss, val_loss = load_model(model_name, loss_name)
            logging.info(f"Model loaded from {model_name}, loss loaded from {loss_name}, trained for {len(train_loss)} epochs")
            return model, train_loss, val_loss
        else:
            model = load_model(model_name)
            logging.info(f"Model loaded from {model_name}, no loss information found")
            return model, [], []
    else:
        return None, [], []

def plot_losses(train_losses, valid_losses, img_path=None):
    plt.figure(figsize=(10, 5))
    # plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    # plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Valid Loss')
    plt.semilogy(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.semilogy(range(1, len(valid_losses) + 1), valid_losses, label='Valid Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs Epochs')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if img_path == None:
        plt.show()
    else:
        plt.savefig(img_path)

def evaluate_model(model, test_loader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Arrays of testing results
    y_true = []
    y_probs = []

    # Iterate the validation set by batches.
    for batch in tqdm(test_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #imgs = imgs.half()

        #Convert label to CPU and numpy
        labels = labels.cpu().numpy()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))
            probs = torch.softmax(logits, dim = 1).cpu().numpy()

        y_true.append(labels)
        y_probs.append(probs)

    # Convert all arrays to numpy results
    y_numpytrue = np.concatenate(y_true, axis=0)
    y_numpyprobs = np.concatenate(y_probs, axis=0)

    # Confirm prediction for accuracy and F1/F2 by using the 50% baseline
    y_numpypred = np.argmax(y_numpyprobs, axis=1)

    # Compute Confusion Matrix
    cm = confusion_matrix(y_numpytrue, y_numpypred)

    # Plot Confusion Matrix
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()

    # Add labels
    plt.xticks([0, 1], ["Human", "AI"])
    plt.yticks([0, 1], ["Human", "AI"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=14)
    plt.show()

    # Extract values from the confusion matrix
    TN, FP, FN, TP = cm.ravel()

    # Compute accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Compute precision and recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Compute F1-score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Compute F2-score
    f2 = 5 * (precision * recall) / ((4 * precision) + recall) if ((4 * precision) + recall) > 0 else 0

    # Compute false positive rate and true positive rate for ROC curve plotting
    fpr, tpr, _ = roc_curve(y_numpytrue, y_numpyprobs[:, 1])

    # Plot the ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='orange', lw=2, label=f"ROC Curve")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve")
    plt.grid(True)
    plt.show()

    # Compute AUC
    auc = roc_auc_score(y_numpytrue, y_numpyprobs[:, 1], multi_class = "ovr")

    # Print results
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F2 Score: {f2:.4f}")
    print(f"AUC: {auc:.4f}")
    return accuracy, f1, f2, auc
