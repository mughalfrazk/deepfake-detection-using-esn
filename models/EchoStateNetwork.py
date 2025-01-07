import numpy as np
import torch

class EchoStateNetwork:
    def __init__(self, reservoir_size, input_dim, spectral_radius=0.9, device="cpu"):
        self.device = torch.device(device)
        self.reservoir_size = reservoir_size

        # Reservoir weights
        self.W_res = (
            torch.rand(reservoir_size, reservoir_size, device=self.device) - 0.5
        )
        eigvals = torch.linalg.eigvals(self.W_res).abs()
        self.W_res *= spectral_radius / eigvals.max()

        # Input weights
        self.W_in = torch.rand(reservoir_size, input_dim, device=self.device) - 0.5

        # Output weights (to be trained)
        self.W_out = None

    def train(self, input_data, target_data):
        # Convert input and target data to tensors
        input_data = torch.tensor(input_data, device=self.device, dtype=torch.float32)
        target_data = torch.tensor(target_data, device=self.device, dtype=torch.float32)

        # Run reservoir with input data
        reservoir_states = self.run_reservoir(input_data)

        # Train the output weights using pseudo-inverse
        self.W_out = torch.matmul(torch.linalg.pinv(reservoir_states), target_data)

    def predict(self, input_data):
        # Convert input data to tensor
        input_data = torch.tensor(input_data, device=self.device, dtype=torch.float32)

        # Run reservoir with input data
        reservoir_states = self.run_reservoir(input_data)

        # Make predictions using the trained output weights
        predictions = torch.matmul(reservoir_states, self.W_out)

        return predictions.cpu().numpy()

    def run_reservoir(self, input_data):
        # Initialize reservoir states
        reservoir_states = torch.zeros(
            (len(input_data), self.reservoir_size), device=self.device
        )

        # Run the reservoir
        for t in range(1, len(input_data)):
            reservoir_states[t, :] = torch.tanh(
                torch.matmul(self.W_res, reservoir_states[t - 1, :])
                + torch.matmul(self.W_in, input_data[t])
            )

        return reservoir_states
