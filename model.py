"""
PyTorch Lightning Autoencoder for dimensionality reduction.

Supports configurable hidden layers, activation, denoising, and
PCA-only fallback mode.
"""

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA


class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int] | None = None,
        latent_dim: int = 2,
        activation: str = "relu",
        denoising_factor: float = 0.0,
        optimizer_name: str = "adam",
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        if hidden_layers is None:
            hidden_layers = [256, 128, 64]

        self.denoising_factor = denoising_factor
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate

        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]

        # Build encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(act_fn())
            in_dim = h_dim
        # Bottleneck has no activation — let values be unbounded
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder (mirror of encoder)
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(act_fn())
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.loss_fn = nn.MSELoss()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def training_step(self, batch, batch_idx):
        (x,) = batch
        if self.denoising_factor > 0 and self.training:
            noisy_x = x + self.denoising_factor * torch.randn_like(x)
        else:
            noisy_x = x
        x_hat = self(noisy_x)
        loss = self.loss_fn(x_hat, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class LossHistoryCallback(pl.Callback):
    """Records per-epoch training loss for live chart display."""

    def __init__(self, loss_history: list | None = None, progress_callback=None):
        super().__init__()
        self.loss_history = loss_history if loss_history is not None else []
        self.progress_callback = progress_callback

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.loss_history.append(float(loss))
        if self.progress_callback is not None:
            self.progress_callback(trainer.current_epoch, trainer.max_epochs)


def train_autoencoder(
    X: np.ndarray,
    config: dict,
    loss_history: list | None = None,
    progress_callback=None,
) -> tuple[Autoencoder, np.ndarray]:
    """
    Train the autoencoder and return the model + 2D embeddings.

    Config keys:
        hidden_layers, latent_dim, activation, denoising_factor,
        optimizer_name, learning_rate, batch_size, max_epochs, patience
    """
    input_dim = X.shape[1]
    dataset = TensorDataset(torch.from_numpy(X).float())
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 64),
        shuffle=True,
        num_workers=0,
    )

    model = Autoencoder(
        input_dim=input_dim,
        hidden_layers=config.get("hidden_layers", [256, 128, 64]),
        latent_dim=config.get("latent_dim", 2),
        activation=config.get("activation", "relu"),
        denoising_factor=config.get("denoising_factor", 0.0),
        optimizer_name=config.get("optimizer_name", "adam"),
        learning_rate=config.get("learning_rate", 1e-3),
    )

    if loss_history is None:
        loss_history = []

    history_cb = LossHistoryCallback(loss_history, progress_callback)
    early_stop = pl.callbacks.EarlyStopping(
        monitor="train_loss",
        patience=config.get("patience", 10),
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=config.get("max_epochs", 200),
        callbacks=[history_cb, early_stop],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="auto",
    )

    trainer.fit(model, dataloader)

    # Extract embeddings
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float()
        embeddings = model.encode(X_tensor).numpy()

    return model, embeddings


def apply_pca_orthogonalization(embeddings: np.ndarray) -> np.ndarray:
    """Apply PCA to the 2D latent space to force orthogonality."""
    return PCA(n_components=2).fit_transform(embeddings)


def run_pca_only(X: np.ndarray) -> np.ndarray:
    """PCA-only mode: project full feature matrix to 2D directly."""
    return PCA(n_components=2).fit_transform(X)


if __name__ == "__main__":
    from data import load_and_preprocess

    X, df, pipe = load_and_preprocess("pokemon.csv", ["Base Stats"])
    print(f"Input shape: {X.shape}")

    config = {
        "hidden_layers": [64, 32],
        "max_epochs": 10,
        "batch_size": 64,
        "patience": 5,
    }
    loss_history = []
    model, embeddings = train_autoencoder(X, config, loss_history)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Final loss: {loss_history[-1]:.4f}")
    print(f"Loss history: {[f'{l:.4f}' for l in loss_history]}")

    # Test PCA modes
    emb_pca = apply_pca_orthogonalization(embeddings)
    print(f"PCA-orthogonalized shape: {emb_pca.shape}")

    emb_pca_only = run_pca_only(X)
    print(f"PCA-only shape: {emb_pca_only.shape}")
