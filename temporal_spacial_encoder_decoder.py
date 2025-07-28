import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
from util import generate_checksum
from temporal_encoder import *
from spacial_encoder import *
from unet_decoder import *
import matplotlib.pyplot as plt
import os
import random

class FullModel(nn.Module):
    def __init__(self, temp_config={}, spacial_config={}, decoder_config={}):
        super().__init__()
        self.temporal_encoder = Temporal_Encoder(**temp_config)
        self.spacial_encoder = Spacial_Encoder(**spacial_config)
        decoder_config['bottleneck_channels'] = self.spacial_encoder.output_channels
        decoder_config['use_skip'] = spacial_config.get('use_skip', True)
        self.decoder = Unet_Decoder(**decoder_config)

    def forward(self, temporal_input, extra_input):
        temp_out = self.temporal_encoder(temporal_input)
        x = torch.cat([temp_out, extra_input], dim=1)
        latent, skips = self.spacial_encoder(x)
        return self.decoder(latent, skips)

def train_temporal_spacial_model(model, optimizer, train_loader, valid_loader,
    n_epochs=50, patience=10, criterion=nn.MSELoss(), output_dir=".",
    train_losses=[], valid_losses=[], prefix="TemporalSpacial"):

    stale = 0
    best_MSE = 1e+9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _train(epoch):
        model.train()
        train_loss = []

        for batch in tqdm(train_loader):
            feature, label = batch
            # temporal_input = feature[:, :20].to(device)
            # extra_input = feature[:, 20:].to(device)
            temporal_input = feature[:, 4:].to(device)
            extra_input = feature[:, :4].to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(temporal_input, extra_input)
            loss = criterion(output, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            train_loss.append(loss.item())

        train_loss = sum(train_loss) / len(train_loss)
        train_losses.append(train_loss)
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5g}")

    def _validate(epoch, record=True):
        model.eval()
        valid_loss = []
        nonlocal best_MSE

        for batch in tqdm(valid_loader):
            imgs, labels = batch
            # temporal_input = imgs[:, :20].to(device)
            # extra_input = imgs[:, 20:].to(device)
            temporal_input = imgs[:, 4:].to(device)
            extra_input = imgs[:, :4].to(device)
            labels = labels.to(device)
            with torch.no_grad():
                logits = model(temporal_input, extra_input)
            loss = criterion(logits, labels)
            valid_loss.append(loss.item())

        valid_loss = sum(valid_loss) / len(valid_loss)
        if record:
            valid_losses.append(valid_loss)
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5g}")

        with open(f"./{prefix}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5g}{' -> best' if valid_loss < best_MSE else ''}")

        return valid_loss

    if len(train_losses) > 0:
        raise RuntimeError(f"len(train_losses) = {len(train_losses)}")
        epoch = 0
        best_MSE = _validate(epoch=0, record=False)

    for epoch in range(len(train_losses), n_epochs):
        _train(epoch)
        val_loss = _validate(epoch)

        if val_loss < best_MSE:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(), f"{output_dir}/{prefix}_best.ckpt")
            with open(f"{output_dir}/{prefix}_best.sum", "w") as fsum:
                fsum.write(generate_checksum(f"{output_dir}/{prefix}_best.ckpt"))
            best_MSE = val_loss
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvement {patience} consecutive epochs, early stopping")
                break

        with open(f"{output_dir}/{prefix}_loss.pkl", "wb") as fout:
            pickle.dump({"train": train_losses, "validation": valid_losses}, fout)
    return train_losses, valid_losses

def load_model_TESD(model_path, temp_config, spacial_config, decoder_config, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = FullModel(temp_config, spacial_config, decoder_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate_single_sample_TESD(model, valid_loader, output_path='comparison.png', device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batches = list(valid_loader)
    batch = random.choice(batches)
    
    feature_batch, label_batch = batch
    idx = random.randint(0, feature_batch.size(0) - 1)

    feature = feature_batch[idx].unsqueeze(0).to(device)  # Shape: (1, C, H, W)
    label = label_batch[idx].unsqueeze(0).to(device)      # Shape: (1, 1, H, W)

    temporal_input = feature[:, 4:]  # (1, T, H, W)
    extra_input = feature[:, :4]     # (1, 4, H, W)

    model.eval()
    with torch.no_grad():
        output = model(temporal_input, extra_input)

    pred_img = output[0].squeeze().cpu().numpy()
    true_img = label[0].squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(true_img, cmap='gray')
    axs[0].set_title('Ground Truth')
    axs[0].axis('off')

    axs[1].imshow(pred_img, cmap='gray')
    axs[1].set_title('Model Output')
    axs[1].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Comparison image saved to {output_path}")
