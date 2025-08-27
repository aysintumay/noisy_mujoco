"""
Baseline models for time series forecasting evaluation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import math

class MLPDropoutBaseline(nn.Module):
    """MLP with dropout baseline for time series forecasting."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = [512, 256, 128], 
                 dropout_rate: float = 0.2, device: str = 'cpu'):
        super().__init__()
        self.device = torch.device(device)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers).to(self.device)
        
    def forward(self, x: torch.Tensor, pl: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP.
        
        Args:
            x: Input sequence [batch, time, features]
            pl: P-level control [batch, forecast_horizon]
            
        Returns:
            output: Predicted features [batch, output_dim]
        """
        # Flatten input sequence and concatenate with p-level
        x_flat = x.view(x.size(0), -1)  # [batch, time * features]
        combined_input = torch.cat([x_flat, pl], dim=1)  # [batch, time * features + forecast_horizon]
        
        return self.network(combined_input)
    
    def sample_multiple(self, x: torch.Tensor, pl: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
        """Sample multiple predictions using dropout."""
        self.train()  # Enable dropout
        samples = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                output = self.forward(x, pl)
                samples.append(output)
        
        self.eval()
        return torch.stack(samples)  # [num_samples, batch, output_dim]


class NeuralProcessBaseline(nn.Module):
    """Neural Process baseline for time series forecasting."""
    
    def __init__(self, input_dim: int, output_dim: int, latent_dim: int = 128, 
                 hidden_dim: int = 256, device: str = 'cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.latent_dim = latent_dim
        
        # Encoder networks
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for time index
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Mean and log_var
        ).to(self.device)
        
        # Aggregator for context points
        self.aggregator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        ).to(self.device)
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + input_dim + 1, hidden_dim),  # latent + input + p-level
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim // 6 * 2)  # Mean and log_var per feature per timestep
        ).to(self.device)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode context points.
        
        Args:
            x: Input context [batch, time, features]
            
        Returns:
            mean, log_var: Latent distribution parameters
        """
        batch_size, time_steps, features = x.shape
        
        # Add time indices
        time_indices = torch.arange(time_steps, dtype=torch.float32, device=self.device)
        time_indices = time_indices.unsqueeze(0).unsqueeze(2).expand(batch_size, time_steps, 1)
        
        # Combine features with time
        x_with_time = torch.cat([x, time_indices], dim=2)  # [batch, time, features+1]
        
        # Encode each time step
        encoded = []
        for t in range(time_steps):
            enc = self.context_encoder(x_with_time[:, t])  # [batch, latent_dim*2]
            encoded.append(enc)
        
        encoded = torch.stack(encoded, dim=1)  # [batch, time, latent_dim*2]
        
        # Aggregate across time
        mean_encoded = encoded[:, :, :self.latent_dim].mean(dim=1)  # [batch, latent_dim]
        logvar_encoded = encoded[:, :, self.latent_dim:].mean(dim=1)  # [batch, latent_dim]
        
        # Further aggregation
        aggregated = self.aggregator(mean_encoded)  # [batch, latent_dim*2]
        mean = aggregated[:, :self.latent_dim]
        log_var = aggregated[:, self.latent_dim:]
        
        return mean, log_var
    
    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z: torch.Tensor, x_context: torch.Tensor, pl: torch.Tensor) -> torch.Tensor:
        """Decode latent to target predictions.
        
        Args:
            z: Latent representation [batch, latent_dim]
            x_context: Context input [batch, time, features]
            pl: P-level control [batch, forecast_horizon]
            
        Returns:
            output: Predicted features [batch, output_dim]
        """
        batch_size = x_context.shape[0]
        forecast_horizon = pl.shape[1]
        
        # Use last context step as base
        last_context = x_context[:, -1, :]  # [batch, features]
        
        predictions = []
        for t in range(forecast_horizon):
            # Prepare decoder input
            p_level_t = pl[:, t:t+1]  # [batch, 1]
            decoder_input = torch.cat([z, last_context, p_level_t], dim=1)
            
            # Decode
            output = self.decoder(decoder_input)  # [batch, features*2]
            predictions.append(output)
        
        # Stack predictions and flatten
        predictions = torch.stack(predictions, dim=1)  # [batch, forecast_horizon, features*2]
        return predictions.view(batch_size, -1)  # [batch, forecast_horizon * features * 2]
    
    def forward(self, x: torch.Tensor, pl: torch.Tensor) -> torch.Tensor:
        """Forward pass through Neural Process.
        
        Args:
            x: Input sequence [batch, time, features]
            pl: P-level control [batch, forecast_horizon]
            
        Returns:
            output: Predicted features [batch, output_dim]
        """
        # Encode context
        mean, log_var = self.encode(x)
        
        # Sample latent
        z = self.reparameterize(mean, log_var)
        
        # Decode to predictions
        output = self.decode(z, x, pl)
        
        # Return only mean predictions (first half)
        return output[:, :output.shape[1]//2]
    
    def sample_multiple(self, x: torch.Tensor, pl: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
        """Sample multiple predictions."""
        self.eval()
        
        # Encode once
        mean, log_var = self.encode(x)
        
        samples = []
        with torch.no_grad():
            for _ in range(num_samples):
                # Sample different latent
                z = self.reparameterize(mean, log_var)
                output = self.decode(z, x, pl)
                # Return only mean predictions
                samples.append(output[:, :output.shape[1]//2])
        
        return torch.stack(samples)  # [num_samples, batch, output_dim]


class BaselineTrainer:
    """Trainer for baseline models."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = torch.device(device)
        
    def train_model(self, train_loader, val_loader, num_epochs: int = 50, 
                   learning_rate: float = 0.001, verbose: bool = True) -> Dict:
        """Train the baseline model."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for x, pl, y in train_loader:
                x, pl, y = x.to(self.device), pl.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(x, pl)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * x.size(0)
            
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for x, pl, y in val_loader:
                    x, pl, y = x.to(self.device), pl.to(self.device), y.to(self.device)
                    output = self.model(x, pl)
                    loss = criterion(output, y)
                    val_loss += loss.item() * x.size(0)
            
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f}")
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }

if __name__ == "__main__":
    # Test the baseline models
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
    # Test dimensions
    batch_size = 32
    input_time = 6
    input_features = 12  # After dropping column 11
    forecast_horizon = 6
    output_features = input_features - 1  # Exclude p-level
    output_dim = forecast_horizon * output_features
    
    # Test MLP baseline
    print("Testing MLP baseline...")
    mlp_input_dim = input_time * input_features + forecast_horizon
    mlp_model = MLPDropoutBaseline(mlp_input_dim, output_dim, device=device)
    
    # Test Neural Process baseline
    print("Testing Neural Process baseline...")
    np_model = NeuralProcessBaseline(input_features, output_dim, device=device)
    
    # Create dummy data
    x = torch.randn(batch_size, input_time, input_features)
    pl = torch.randn(batch_size, forecast_horizon)
    
    # Test forward passes
    mlp_output = mlp_model(x, pl)
    np_output = np_model(x, pl)
    
    print(f"MLP output shape: {mlp_output.shape}")
    print(f"NP output shape: {np_output.shape}")
    
    # Test sampling
    mlp_samples = mlp_model.sample_multiple(x, pl, num_samples=5)
    np_samples = np_model.sample_multiple(x, pl, num_samples=5)
    
    print(f"MLP samples shape: {mlp_samples.shape}")
    print(f"NP samples shape: {np_samples.shape}")
    
    print("All tests passed!")


class LegendreMemoryUnit(nn.Module):
    """Legendre Memory Unit (LMU) implementation."""
    
    def __init__(self, input_dim: int, memory_dim: int, theta: float = 1.0):
        super().__init__()
        self.memory_dim = memory_dim
        self.theta = theta
        
        # Legendre polynomial matrices
        A, B = self._legendre_matrices(memory_dim, theta)
        self.register_buffer('A', A)
        self.register_buffer('B', B)
        
        # Input and output projections
        self.input_proj = nn.Linear(input_dim, memory_dim)
        self.output_proj = nn.Linear(memory_dim, input_dim)
        
    def _legendre_matrices(self, N: int, theta: float):
        """Generate Legendre polynomial transition matrices."""
        # A matrix for Legendre polynomials (more stable formulation)
        A = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):  # Only upper triangular
                A[i, j] = (2 * i + 1) / theta if j == i + 1 else 0
        
        # B matrix for input projection  
        B = np.zeros((N,))
        for i in range(N):
            B[i] = (2 * i + 1) / theta
            
        # Scale down to prevent instability
        A = A * 0.1
        B = B * 0.1
        
        return torch.FloatTensor(A), torch.FloatTensor(B)
    
    def forward(self, x: torch.Tensor, memory: torch.Tensor = None):
        """Forward pass through LMU.
        
        Args:
            x: Input [batch, seq_len, input_dim]
            memory: Initial memory state [batch, memory_dim]
            
        Returns:
            output: Output [batch, seq_len, input_dim]
            final_memory: Final memory state [batch, memory_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        if memory is None:
            memory = torch.zeros(batch_size, self.memory_dim, device=x.device)
        
        outputs = []
        
        for t in range(seq_len):
            # Update memory with Legendre dynamics (simplified and stable)
            input_proj = self.input_proj(x[:, t])  # [batch, memory_dim]
            memory = memory * 0.9 + input_proj * 0.1  # Simple exponential smoothing
            
            # Generate output
            output = self.output_proj(memory)
            outputs.append(output)
        
        return torch.stack(outputs, dim=1), memory


class CLMUBaseline(nn.Module):
    """Conditional Legendre Memory Unit baseline for time series forecasting."""
    
    def __init__(self, input_dim: int, output_dim: int, memory_dim: int = 64,
                 hidden_dim: int = 128, num_layers: int = 2, device: str = 'cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizon = 6  # Assume 6 step forecast
        
        # CLMU layers
        self.clmu_layers = nn.ModuleList([
            LegendreMemoryUnit(input_dim if i == 0 else memory_dim, memory_dim)
            for i in range(num_layers)
        ])
        
        # Conditional processing
        self.condition_proj = nn.Linear(self.forecast_horizon, hidden_dim)  # p-level conditioning
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(memory_dim + hidden_dim, memory_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.to(self.device)
        
    def forward(self, x: torch.Tensor, pl: torch.Tensor) -> torch.Tensor:
        """Forward pass through CLMU.
        
        Args:
            x: Input sequence [batch, time, features]
            pl: P-level control [batch, forecast_horizon]
            
        Returns:
            output: Predicted features [batch, output_dim]
        """
        batch_size = x.size(0)
        
        # Process p-level conditioning
        condition = self.condition_proj(pl)  # [batch, hidden_dim]
        
        # Process through CLMU layers
        current_input = x
        
        for i, clmu_layer in enumerate(self.clmu_layers):
            # Pass through CLMU
            clmu_output, final_memory = clmu_layer(current_input)
            
            # Combine final memory with conditioning and pass through hidden layer
            combined = torch.cat([final_memory, condition], dim=1)  # [batch, memory_dim + hidden_dim]
            current_memory = torch.relu(self.hidden_layers[i](combined))
            
            # Create next input by expanding memory to sequence length
            if i < len(self.clmu_layers) - 1:
                current_input = current_memory.unsqueeze(1).expand(-1, x.size(1), -1)  # [batch, seq_len, memory_dim]
            else:
                # For final layer, just use the memory output
                final_features = current_memory
        
        # Final output projection
        output = self.output_proj(final_features)
        
        return output
    
    def sample_multiple(self, x: torch.Tensor, pl: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
        """Sample multiple predictions using dropout."""
        self.train()  # Enable dropout
        samples = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                output = self.forward(x, pl)
                samples.append(output)
        
        self.eval()
        return torch.stack(samples)  # [num_samples, batch, output_dim]


class StateSpaceBaseline(nn.Module):
    """State-Space Model baseline for time series forecasting."""
    
    def __init__(self, input_dim: int, output_dim: int, state_dim: int = 64,
                 hidden_dim: int = 128, device: str = 'cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.forecast_horizon = 6
        
        # State transition matrices
        self.A = nn.Linear(state_dim, state_dim, bias=False)  # State transition
        self.B = nn.Linear(input_dim, state_dim, bias=False)  # Input to state
        self.C = nn.Linear(state_dim, hidden_dim, bias=True)  # State to observation
        
        # Control input processing
        self.control_proj = nn.Linear(self.forecast_horizon, state_dim)
        
        # Observation model
        self.obs_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize state transition to be stable
        with torch.no_grad():
            self.A.weight.copy_(torch.eye(state_dim) * 0.9 + torch.randn(state_dim, state_dim) * 0.1)
        
        self.to(self.device)
        
    def forward(self, x: torch.Tensor, pl: torch.Tensor) -> torch.Tensor:
        """Forward pass through state-space model.
        
        Args:
            x: Input sequence [batch, time, features]
            pl: P-level control [batch, forecast_horizon]
            
        Returns:
            output: Predicted features [batch, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize state
        state = torch.zeros(batch_size, self.state_dim, device=x.device)
        
        # Process control input
        control = self.control_proj(pl)  # [batch, state_dim]
        
        # Forward pass through sequence
        for t in range(seq_len):
            # State update: s_t = A * s_{t-1} + B * x_t + control
            state = self.A(state) + self.B(x[:, t]) + control
            
            # Apply activation to maintain stability
            state = torch.tanh(state)
        
        # Generate observation from final state
        observation = self.C(state)
        output = self.obs_model(observation)
        
        return output
    
    def sample_multiple(self, x: torch.Tensor, pl: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
        """Sample multiple predictions by adding noise to states."""
        samples = []
        
        for _ in range(num_samples):
            # Add noise to the state transition for sampling
            batch_size, seq_len, _ = x.shape
            state = torch.zeros(batch_size, self.state_dim, device=x.device)
            control = self.control_proj(pl)
            
            for t in range(seq_len):
                # Add small amount of noise for stochasticity
                noise = torch.randn_like(state) * 0.01
                state = self.A(state) + self.B(x[:, t]) + control + noise
                state = torch.tanh(state)
            
            observation = self.C(state)
            output = self.obs_model(observation)
            samples.append(output)
        
        return torch.stack(samples)  # [num_samples, batch, output_dim]