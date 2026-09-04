from __future__ import annotations

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    Generic LSTM-based binary sequence classifier.

    Expected input shape:
        (batch_size, sequence_length, input_size)

    Output shape:
        (batch_size,)

    Output values:
        Probability between 0 and 1.

    This model is dataset-independent.
    It can be used with any numerical sequential dataset
    as long as the input is converted to the expected shape.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Validate configuration
        # ---------------------------------------------------------

        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError(
                "input_size must be a positive integer."
            )

        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(
                "hidden_size must be a positive integer."
            )

        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError(
                "num_layers must be a positive integer."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0, 1)."
            )

        # ---------------------------------------------------------
        # LSTM
        # ---------------------------------------------------------

        # PyTorch only applies dropout between LSTM layers.
        # Therefore, disable internal LSTM dropout when num_layers=1.
        lstm_dropout = (
            dropout if num_layers > 1 else 0.0
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # ---------------------------------------------------------
        # External dropout
        # ---------------------------------------------------------

        self.dropout = nn.Dropout(
            p=dropout
        )

        # ---------------------------------------------------------
        # Classification head
        # ---------------------------------------------------------

        self.fc = nn.Linear(
            hidden_size,
            1,
        )

        # ---------------------------------------------------------
        # Probability conversion
        # ---------------------------------------------------------

        self.sigmoid = nn.Sigmoid()

        # Store configuration for reproducibility/debugging.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Tensor with shape:

                (batch, sequence_length, features)

        Returns
        -------
        torch.Tensor
            Probability for the positive class with shape:

                (batch,)
        """

        # ---------------------------------------------------------
        # Input validation
        # ---------------------------------------------------------

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "Input must be a torch.Tensor."
            )

        if x.ndim != 3:
            raise ValueError(
                "Expected input with 3 dimensions "
                "(batch, sequence_length, features), "
                f"but received shape {tuple(x.shape)}."
            )

        if x.size(-1) != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, "
                f"but received {x.size(-1)}."
            )

        # ---------------------------------------------------------
        # LSTM
        # ---------------------------------------------------------

        output, _ = self.lstm(x)

        # ---------------------------------------------------------
        # Take the final timestep
        # ---------------------------------------------------------

        last_output = output[:, -1, :]

        # ---------------------------------------------------------
        # Dropout
        # ---------------------------------------------------------

        last_output = self.dropout(
            last_output
        )

        # ---------------------------------------------------------
        # Fully connected layer
        # ---------------------------------------------------------

        logits = self.fc(
            last_output
        )

        # ---------------------------------------------------------
        # Convert logits -> probability
        # ---------------------------------------------------------

        probabilities = self.sigmoid(
            logits
        )

        # Shape:
        # (batch, 1) -> (batch,)
        probabilities = probabilities.squeeze(
            -1
        )

        return probabilities

    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Generate binary predictions.

        Parameters
        ----------
        x:
            Input tensor with shape:

                (batch, sequence_length, features)

        threshold:
            Classification threshold between 0 and 1.

        Returns
        -------
        torch.Tensor
            Binary predictions with shape:

                (batch,)
        """

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        self.eval()

        with torch.no_grad():
            probabilities = self.forward(x)

        return (
            probabilities >= threshold
        ).long()

    def predict_proba(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return positive-class probabilities.

        This is useful during evaluation and inference.
        """

        self.eval()

        with torch.no_grad():
            probabilities = self.forward(x)

        return probabilities

    def get_config(self) -> dict:
        """
        Return model configuration.

        Useful for saving model metadata.
        """

        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
        }