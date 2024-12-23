import numpy as np

class EchoStateNetwork:
  def __init__(self, reservoir_size, input_dim, spectral_radius=0.9):
    # Initialize network parameters
    self.reservoir_size = reservoir_size

    # Reservoir weights
    self.W_res = np.random.rand(reservoir_size, reservoir_size) - 0.5
    self.W_res *= spectral_radius / np.max(np.abs(np.linalg.eigvals(self.W_res)))

    # Input weights
    self.W_in = np.random.rand(reservoir_size, input_dim) - 0.5

    # Output weights (to be trained)
    self.W_out = None

  def train(self, input_data, target_data):
    # Run reservoir with input data
    reservoir_states = self.run_reservoir(input_data)

    # Train the output weights using pseudo-inverse
    self.W_out = np.dot(np.linalg.pinv(reservoir_states), target_data)

  def predict(self, input_data):
    # Run reservoir with input data
    reservoir_states = self.run_reservoir(input_data)

    # Make predictions using the trained output weights
    predictions = np.dot(reservoir_states, self.W_out)

    return predictions

  def run_reservoir(self, input_data):
    # Initialize reservoir states
    reservoir_states = np.zeros((len(input_data), self.reservoir_size))

    # Run the reservoir
    for t in range(1, len(input_data)):
        reservoir_states[t, :] = np.tanh(
            np.dot(self.W_res, reservoir_states[t - 1, :])
            + np.dot(self.W_in, input_data[t])
        )

    return reservoir_states
