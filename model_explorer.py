import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from temporal_encoder import *
from spacial_encoder import *
from unet_decoder import *
from temporal_spacial_encoder_decoder import *
import numpy as np
from util import *
from data_loader import data_loader
from sklearn.model_selection import train_test_split
from copy import deepcopy
import time
import os, sys
from CNN import plot_losses

class ML_explorer:
    def __init__(self, train_loader, valid_loader, output_dir, train_config):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_config = train_config
        self.output_dir = output_dir
        self.min_loss = np.inf

    def load_min_loss_from_study(self, study):
        self.iteration = len(study.trials)
        if len(study.trials) != 0:
            self.min_loss = study.best_value
            logging.info(f"Best trial value from study: {self.min_loss}")
        return self.min_loss

class TSED_explorer(ML_explorer):
    def __init__(self, train_loader, valid_loader, output_dir, train_config):
        super().__init__(train_loader, valid_loader, output_dir, train_config)
    def __call__(self, trial):
        base_features = trial.suggest_categorical("base_features", [32, 64, 128, 256])
        temporal_encoder_config = {
            "embed_dim": trial.suggest_categorical("temporal_encoder.embed_dim", [128, 256, 512, 1024, 2048]),
            "num_heads": trial.suggest_categorical("temporal_encoder.num_heads", [2, 4, 8, 16, 32, 64]),
            "num_layers": trial.suggest_int("temporal_encoder.num_layers", 2, 6),
            "mlp_ratio": trial.suggest_float("temporal_encoder.mlp_ratio", 1.0, 7.0),
        }
        spacial_encoder_config = {
            "base_features": base_features,
            "num_layers": trial.suggest_int("spacial_encoder.num_layers", 1, 4),
        }

        s_layers = spacial_encoder_config["num_layers"]

        decoder_config = {
            "out_channels": 1,
            "num_layers": s_layers,
            "bottleneck_channels": base_features * (2 ** s_layers),
        }
        model = FullModel(temp_config = temporal_encoder_config,
            spacial_config = spacial_encoder_config,
            decoder_config = decoder_config)
        self.train_config["train_losses"] = []
        self.train_config["valid_losses"] = []
        self.train_config["output_dir"] = f"{self.output_dir}/local"
        os.makedirs(f"{self.output_dir}/local", exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model.to(device)
        except:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            model.to(device)
        criterion = nn.MSELoss()
        # optimizer = optim.Adam(model.parameters(), lr=trial.suggest_float("learning_rate", 5e-5, 1e-2, log=True))
        optimizer = optim.Adam(model.parameters(), lr=4e-3)

        train_losses, valid_losses = train_temporal_spacial_model(model, optimizer, self.train_loader, self.valid_loader, criterion = criterion, **self.train_config)
        ret = min(valid_losses)

        if ret < self.min_loss:
            self.min_loss = ret
            prefix = self.train_config["prefix"]
            model_path = f"{self.output_dir}/local/{prefix}_best.ckpt"
            model.load_state_dict(torch.load(model_path, map_location=device))
            torch.save(model.state_dict(), f"{self.output_dir}/best_model.ckpt")
            with open(f"{self.output_dir}/best_hyperparams.pkl", "wb") as fout:
                pickle.dump((temporal_encoder_config, spacial_encoder_config, decoder_config), fout)

            plot_losses(train_losses, valid_losses, f"{self.output_dir}/loss.png")
            logging.info(f"Saved best model: val_loss = {min(valid_losses)}, train_loss = {min(train_losses)}")

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return ret

def run(output_dir, data_dir, prefix, n_trials=100, epochs=20, **kwargs):
    train_config = {
        "n_epochs": epochs,
        "patience": 10,
        "prefix": prefix,
    }

    _all_dataset = data_loader(os.path.join(data_dir, "feature"), os.path.join(data_dir, "label"))
    _files = _all_dataset.file_names
    train_files, valid_files = train_test_split(_files, test_size=.2)
    train_dataset = deepcopy(_all_dataset)
    train_dataset.file_names = train_files
    valid_dataset = deepcopy(_all_dataset)
    valid_dataset.file_names = valid_files


    train_loader = train_dataset.get_torch_loader()
    valid_loader = valid_dataset.get_torch_loader()

    t_start = time.time()
    explorer = TSED_explorer(train_loader, valid_loader, output_dir, train_config)
    os.makedirs(output_dir, exist_ok=True)

    study = optuna.create_study(
        direction="minimize",
        study_name=prefix,
        storage=f"sqlite:///{output_dir}/study.db",
        load_if_exists=True,
    )
    explorer.load_min_loss_from_study(study)
    try:
        study.optimize(explorer, n_trials=n_trials)
    except KeyboardInterrupt:
        logging.info("Optimization interrupted. Saving the study.")
        with open(f"{output_dir}/study.pkl", "wb") as fout:
            pickle.dump(study, fout)

    logging.info(f"Best hyperparameters: {study.best_params}")
    t_end = time.time()
    logging.info("Total time: {:.2f} seconds".format(t_end - t_start))


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("-n", "--n_trials", type=int, default=100,
        help = "Number of trials for hyperparameter optimization")
    parser.add_argument("-e", "--epochs", type=int, default=20,
        help="Number of epochs for training the model")
    parser.add_argument("-o", "--output_dir", required=True,
        help = "Directory to save the model outputs")
    parser.add_argument("-d", "--data_dir", required=True,
        help = "Directory path to the data files")
    parser.add_argument("-p", "--prefix", required=True,
        help = "Name for this training")

    args = parser.parse_args()
    setup_logger(args)
    run(**args)
