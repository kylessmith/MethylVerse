import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from sklearn.model_selection import train_test_split

class BetaVAE(nn.Module):
    """
    Variational Autoencoder optimized for beta-distributed methylation data.
    """
    def __init__(self, input_dim, latent_dim=10, hidden_dim=128, dropout_rate=0.2):
        super(BetaVAE, self).__init__()
        
        # Same encoder as before
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder with a modified final activation
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_hidden = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.decoder_output = nn.Linear(hidden_dim, input_dim)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
    
    def encode(self, x):
        """
        Encode input data to latent space parameters.
        """
        # Check for NaN inputs
        if torch.isnan(x).any():
            print("Warning: NaN detected in input data")
            x = torch.nan_to_num(x, nan=0.5)
            
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from latent space.
        """
        # Clamp logvar to prevent numerical issues
        logvar = torch.clamp(logvar, -10.0, 10.0)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        Decode latent space representation to output space.
        Using a custom-bounded sigmoid to avoid edge cases.
        """
        z = self.decoder_input(z)
        z = self.decoder_hidden(z)
        x_logits = self.decoder_output(z)
        
        # Custom bounded sigmoid to avoid exact 0 and 1
        # Sigmoid normally maps to (0,1), this maps to (0.001, 0.999)
        x_recon = 0.998 * torch.sigmoid(x_logits) + 0.001
        
        return x_recon
    
    def forward(self, x):
        """
        Forward pass through the VAE.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def generate(self, z):
        """
        Generate samples from latent vectors.
        """
        return self.decode(z)


def beta_loss_function(recon_x, x, mu, logvar, kl_weight=0.1):
    """
    Extremely robust loss function for beta-distributed data.
    Uses MSE for reconstruction to avoid log-based numerical issues entirely.
    
    Parameters:
    -----------
    recon_x : torch.Tensor
        Reconstructed values from decoder (predicted)
    x : torch.Tensor
        Original input values (ground truth)
    mu : torch.Tensor
        Mean of the latent distribution
    logvar : torch.Tensor
        Log variance of the latent distribution
    kl_weight : float
        Weight for the KL divergence term
        
    Returns:
    --------
    total_loss : torch.Tensor
        Combined loss (reconstruction + KL divergence)
    recon_loss : torch.Tensor
        Reconstruction loss component
    kl_loss : torch.Tensor
        KL divergence loss component
    """
    # Instead of using binary cross-entropy which involves log operations,
    # use mean squared error which is numerically stable
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # For KL divergence, use a numerically stable implementation
    # with aggressive clamping and handling of edge cases
    try:
        logvar = torch.clamp(logvar, -10.0, 10.0)  # Prevent extreme values
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Handle potential NaN or inf in KL loss
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            print("Warning: NaN or Inf in KL loss, using fallback")
            kl_loss = torch.tensor(0.1, device=recon_x.device)
    except Exception as e:
        print(f"Error in KL loss calculation: {e}")
        kl_loss = torch.tensor(0.1, device=recon_x.device)
    
    # Combined loss with weight on KL term
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss


def train_beta_vae(df, classes, test_size=0.2, latent_dim=10, hidden_dim=128, 
                  epochs=50, batch_size=32, device=None, learning_rate=1e-4,
                  gradient_clip=1.0, early_stop_patience=5, beta_weight_start=0.0,
                  beta_weight_end=0.1, beta_warmup_epochs=10):
    """
    Train a beta-optimized VAE on methylation data with improved stability.
    Includes beta-VAE warmup to prevent posterior collapse.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with features as rows and samples as columns
    classes : array-like
        Array of class labels for each sample
    test_size : float
        Proportion of data to use for testing
    latent_dim : int
        Dimension of the latent space
    hidden_dim : int
        Dimension of the hidden layers
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    device : torch.device
        Device to run the model on
    learning_rate : float
        Learning rate for optimizer
    gradient_clip : float
        Maximum gradient norm
    early_stop_patience : int
        Number of epochs to wait before early stopping
    beta_weight_start : float
        Starting weight for KL divergence term
    beta_weight_end : float
        Final weight for KL divergence term
    beta_warmup_epochs : int
        Number of epochs over which to linearly increase beta
        
    Returns:
    --------
    model : BetaVAE
        Trained VAE model
    col_means : numpy.ndarray
        Column means used for imputation
    device : torch.device
        Device used for training
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Transpose dataframe to have samples as rows and features as columns
    X = df.T.values
    
    # Check for extreme values in the data
    print(f"Data range: min={np.nanmin(X):.6f}, max={np.nanmax(X):.6f}")
    if np.nanmin(X) < 0 or np.nanmax(X) > 1:
        print("Warning: Data contains values outside the [0,1] range expected for methylation beta values")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, classes, test_size=test_size, stratify=classes, random_state=42
    )
    
    # Handle NaN values in the training data
    print(f"Data contains {np.isnan(X_train).sum()} NaN values out of {X_train.size}")
    
    # Replace NaNs with column means (feature-wise imputation)
    col_means = np.nanmean(X_train, axis=0)
    
    # Find indices of NaN values
    nan_indices = np.isnan(X_train)
    
    # Replace NaNs with column means
    X_train[nan_indices] = np.take(col_means, np.where(nan_indices)[1])
    
    # Do the same for test data
    nan_indices_test = np.isnan(X_test)
    X_test[nan_indices_test] = np.take(col_means, np.where(nan_indices_test)[1])
    
    # Ensure values are in valid range for numerical stability
    # Use a slightly wider margin (1e-6 instead of 0) for stability
    X_train = np.clip(X_train, 1e-6, 1 - 1e-6)
    X_test = np.clip(X_test, 1e-6, 1 - 1e-6)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor)
    test_dataset = TensorDataset(X_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create and initialize the model
    input_dim = X_train.shape[1]
    model = BetaVAE(input_dim, latent_dim, hidden_dim).to(device)
    
    # Use more reliable Xavier uniform initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # Try a more conservative optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler with patience
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, 
        min_lr=1e-6, threshold=1e-4
    )
    
    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        batch_count = 0
        
        # Calculate beta for this epoch (KL weight)
        # This implements beta warmup to prevent posterior collapse
        if epoch < beta_warmup_epochs:
            # Linear warmup
            beta = beta_weight_start + (beta_weight_end - beta_weight_start) * (epoch / beta_warmup_epochs)
        else:
            beta = beta_weight_end
        
        for batch_idx, (data,) in enumerate(train_loader):
            # Check for NaN in input data
            if torch.isnan(data).any():
                print(f"Warning: NaN found in input batch {batch_idx}, replacing with 0.5")
                data = torch.nan_to_num(data, nan=0.5)
                
            # Check for extreme values
            if data.min() < 1e-6 or data.max() > 1 - 1e-6:
                data = torch.clamp(data, 1e-6, 1 - 1e-6)
            
            optimizer.zero_grad()
            
            try:
                # Forward pass
                recon_batch, mu, logvar = model(data)
                
                # Calculate loss with current beta value
                loss, recon_loss, kl_loss = beta_loss_function(
                    recon_batch, data, mu, logvar, kl_weight=beta
                )
                
                # Skip problematic batches
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Skipping batch {batch_idx} due to problematic loss value")
                    continue
                
                # Backward pass and optimization
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                # Check for NaN gradients
                has_nan_grad = False
                for param in model.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print(f"NaN gradient detected in batch {batch_idx}, skipping optimizer step")
                    continue
                    
                optimizer.step()
                
                train_loss += loss.item()
                train_recon_loss += recon_loss.item()
                train_kl_loss += kl_loss.item()
                batch_count += 1
                
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Skip epoch update if all batches were problematic
        if batch_count == 0:
            print(f"Epoch {epoch+1}/{epochs}: All batches were problematic, skipping epoch")
            continue
        
        # Calculate average losses
        avg_train_loss = train_loss / batch_count
        avg_train_recon = train_recon_loss / batch_count
        avg_train_kl = train_kl_loss / batch_count
        
        # Evaluation mode
        model.eval()
        test_loss = 0
        test_batch_count = 0
        
        with torch.no_grad():
            for batch_idx, (data,) in enumerate(test_loader):
                try:
                    # Replace NaNs and clip extreme values
                    if torch.isnan(data).any():
                        data = torch.nan_to_num(data, nan=0.5)
                    data = torch.clamp(data, 1e-6, 1 - 1e-6)
                    
                    recon_batch, mu, logvar = model(data)
                    loss, _, _ = beta_loss_function(recon_batch, data, mu, logvar, kl_weight=beta)
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        test_loss += loss.item()
                        test_batch_count += 1
                except Exception as e:
                    print(f"Error in test batch {batch_idx}: {e}")
                    continue
        
        # Switch back to training mode
        model.train()
        
        # Skip if all test batches were problematic
        if test_batch_count == 0:
            print(f"Epoch {epoch+1}/{epochs}: All test batches were problematic")
            continue
            
        avg_test_loss = test_loss / test_batch_count
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}, Beta: {beta:.4f}, Train Loss: {avg_train_loss:.4f} '
              f'(Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f}), '
              f'Test Loss: {avg_test_loss:.4f}')
        
        # Update learning rate scheduler
        scheduler.step(avg_test_loss)
        
        # Early stopping check
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            patience_counter = 0
            # Save best model
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                # Restore best model
                if best_model_state is not None:
                    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                break
    
    # Final evaluation
    model.eval()
    test_loss = 0
    test_batch_count = 0
    
    with torch.no_grad():
        for batch_idx, (data,) in enumerate(test_loader):
            # Replace NaNs and clip extreme values
            if torch.isnan(data).any():
                data = torch.nan_to_num(data, nan=0.5)
            data = torch.clamp(data, 1e-6, 1 - 1e-6)
            
            recon_batch, mu, logvar = model(data)
            loss, _, _ = beta_loss_function(recon_batch, data, mu, logvar, kl_weight=beta_weight_end)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                test_loss += loss.item()
                test_batch_count += 1
    
    if test_batch_count > 0:
        print(f'Final Test Loss: {test_loss / test_batch_count:.4f}')
    else:
        print("Warning: Could not compute final test loss due to numerical issues")
    
    return model, col_means, device


def apply_sparsity(data, sparsity_level=0.1, sparsity_pattern=None, original_data=None):
    """
    Apply sparsity to the data by randomly setting values to NaN.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Data to apply sparsity to (features as rows)
    sparsity_level : float
        Proportion of values to set to NaN (0.0 to 1.0)
    sparsity_pattern : str, optional
        Type of sparsity pattern to apply:
        - 'random': Completely random (default if None)
        - 'feature_based': Some features are more likely to be missing
        - 'sample_based': Some samples are more likely to be missing
        - 'match_original': Match the sparsity pattern of the original data
    original_data : numpy.ndarray, optional
        Original data to match sparsity pattern to if sparsity_pattern='match_original'
        
    Returns:
    --------
    sparse_data : numpy.ndarray
        Data with NaN values
    """
    # Make a copy to avoid modifying the original
    sparse_data = data.copy()
    
    n_features, n_samples = sparse_data.shape
    total_elements = n_features * n_samples
    n_nans = int(total_elements * sparsity_level)
    
    if sparsity_pattern is None or sparsity_pattern == 'random':
        # Completely random sparsity
        flat_indices = np.random.choice(total_elements, n_nans, replace=False)
        row_indices = flat_indices // n_samples
        col_indices = flat_indices % n_samples
        sparse_data[row_indices, col_indices] = np.nan
        
    elif sparsity_pattern == 'feature_based':
        # Some features are more likely to have missing values
        feature_missing_prob = np.random.beta(2, 5, size=n_features)
        feature_missing_prob = feature_missing_prob / np.sum(feature_missing_prob) * n_nans
        
        for i in range(n_features):
            n_missing = int(feature_missing_prob[i])
            if n_missing > 0:
                missing_cols = np.random.choice(n_samples, min(n_missing, n_samples), replace=False)
                sparse_data[i, missing_cols] = np.nan
                
    elif sparsity_pattern == 'sample_based':
        # Some samples are more likely to have missing values
        sample_missing_prob = np.random.beta(2, 5, size=n_samples)
        sample_missing_prob = sample_missing_prob / np.sum(sample_missing_prob) * n_nans
        
        for j in range(n_samples):
            n_missing = int(sample_missing_prob[j])
            if n_missing > 0:
                missing_rows = np.random.choice(n_features, min(n_missing, n_features), replace=False)
                sparse_data[missing_rows, j] = np.nan
                
    elif sparsity_pattern == 'match_original' and original_data is not None:
        # Match the sparsity pattern of the original data
        original_nan_mask = np.isnan(original_data)
        original_nan_rate = np.mean(original_nan_mask)
        
        # If original has lower sparsity than target, add more NaNs
        if original_nan_rate < sparsity_level:
            additional_rate = sparsity_level - original_nan_rate
            non_nan_mask = ~original_nan_mask
            n_additional_nans = int(additional_rate * total_elements)
            
            # Find valid positions (non-NaN in original)
            valid_positions = np.where(non_nan_mask.flatten())[0]
            
            # Randomly select positions to set to NaN
            if len(valid_positions) > 0:
                selected = np.random.choice(
                    valid_positions, 
                    min(n_additional_nans, len(valid_positions)), 
                    replace=False
                )
                selected_rows = selected // n_samples
                selected_cols = selected % n_samples
                
                # Apply original NaN pattern
                sparse_data[original_nan_mask] = np.nan
                
                # Add additional NaNs
                sparse_data[selected_rows, selected_cols] = np.nan
            
        else:
            # Just use the original pattern (but randomly subsample if needed)
            nan_positions = np.where(original_nan_mask.flatten())[0]
            n_target_nans = int(sparsity_level * total_elements)
            
            if len(nan_positions) > n_target_nans:
                selected = np.random.choice(nan_positions, n_target_nans, replace=False)
                selected_rows = selected // n_samples
                selected_cols = selected % n_samples
                
                # Create a new mask
                new_mask = np.zeros_like(original_nan_mask, dtype=bool)
                new_mask[selected_rows, selected_cols] = True
                sparse_data[new_mask] = np.nan
            else:
                sparse_data[original_nan_mask] = np.nan
    
    return sparse_data

def simulate_methylation_samples_by_class(model, class_to_generate, X, classes, 
                                        n_samples=10, device=None, sparsity_level=0.0,
                                        sparsity_pattern=None, original_data=None):
    """
    Simulate new methylation samples for a specific class with controlled sparsity.
    
    Parameters:
    -----------
    model : BetaVAE
        Trained VAE model
    class_to_generate : any
        The class to generate samples for
    X : numpy.ndarray
        Original data (samples as rows)
    classes : array-like
        Original class labels from training
    n_samples : int
        Number of samples to generate
    device : torch.device
        Device to run the model on
    sparsity_level : float
        Proportion of values to set to NaN (0.0 to 1.0)
    sparsity_pattern : str, optional
        Type of sparsity pattern to apply
    original_data : numpy.ndarray, optional
        Original data to match sparsity pattern to if sparsity_pattern='match_original'
        
    Returns:
    --------
    simulated_samples : numpy.ndarray
        Generated samples with controlled sparsity (features as rows)
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find samples of the target class
    class_indices = np.where(np.array(classes) == class_to_generate)[0]
    
    if len(class_indices) == 0:
        raise ValueError(f"No samples found for class {class_to_generate}")
    
    # Get class-specific samples
    X_class = X[class_indices]
    
    # Convert to PyTorch tensor
    X_class_tensor = torch.FloatTensor(X_class).to(device)
    
    # Encode samples to get latent space distribution
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(X_class_tensor)
        
        # Calculate mean and std in latent space for this class
        z_mean_avg = mu.mean(dim=0)
        z_std = torch.exp(0.5 * logvar).mean(dim=0)
        
        # Generate samples from class-specific distribution
        z_sampled = torch.randn(n_samples, model.latent_dim).to(device)
        z_sampled = z_sampled * z_std + z_mean_avg
        
        # Decode to get new samples
        generated = model.decode(z_sampled).cpu().numpy()
    
    # Ensure valid methylation beta values
    generated = np.clip(generated, 0, 1)
    
    # Transpose to return to original format (features as rows)
    generated = generated.T
    
    # Apply sparsity if requested
    if sparsity_level > 0:
        if sparsity_pattern == 'match_original' and original_data is None:
            # Default to using the original class data if available
            original_class_data = X_class.T  # Transpose to features as rows
            generated = apply_sparsity(generated, sparsity_level, sparsity_pattern, original_class_data)
        else:
            generated = apply_sparsity(generated, sparsity_level, sparsity_pattern, original_data)
    
    return generated

def apply_contamination(simulated_samples, contamination_profiles, contamination_proportion,
                        contamination_method='random_weighted', contamination_params=None):
    """
    Apply contamination to simulated methylation samples by mixing in other profiles.
    
    Parameters:
    -----------
    simulated_samples : numpy.ndarray
        Pure simulated samples (features as rows)
    contamination_profiles : numpy.ndarray or dict
        Profiles to use as contamination sources (features as rows)
        Can be a single array or a dictionary of arrays for multiple sources
    contamination_proportion : float or list
        Overall proportion of contamination to apply (0.0 to 1.0)
        If a list, specifies the proportion for each sample
    contamination_method : str
        Method to apply contamination:
        - 'fixed': Apply the same contamination proportion to all samples
        - 'random': Randomly vary contamination proportion between samples
        - 'random_weighted': Use weighted average with random weights
        - 'beta_distribution': Sample contamination proportions from a beta distribution
    contamination_params : dict, optional
        Additional parameters for the contamination method:
        - For 'beta_distribution': 'alpha' and 'beta' parameters
        - For 'random_weighted': 'weights' for different contamination sources
        
    Returns:
    --------
    contaminated_samples : numpy.ndarray
        Samples with contamination applied (features as rows)
    contamination_levels : numpy.ndarray
        Actual contamination proportion applied to each sample
    """
    # Make a copy to avoid modifying the original
    contaminated_samples = simulated_samples.copy()
    n_features, n_samples = contaminated_samples.shape
    
    # Process contamination_proportion
    if isinstance(contamination_proportion, (int, float)):
        # Single value provided, convert to array
        contamination_proportion = np.ones(n_samples) * contamination_proportion
    elif len(contamination_proportion) != n_samples:
        # List provided but wrong length
        raise ValueError(f"contamination_proportion list length ({len(contamination_proportion)}) " 
                         f"must match number of samples ({n_samples})")
    
    # Initialize contamination levels
    contamination_levels = np.zeros(n_samples)
    
    # Process contamination profiles
    if isinstance(contamination_profiles, dict):
        # Multiple contamination sources
        contamination_sources = list(contamination_profiles.values())
        n_sources = len(contamination_sources)
        
        # Check if all sources have the same number of features
        for source in contamination_sources:
            if source.shape[0] != n_features:
                raise ValueError(f"Contamination source has {source.shape[0]} features, "
                                f"but simulated samples have {n_features}")
    else:
        # Single contamination source
        if contamination_profiles.shape[0] != n_features:
            raise ValueError(f"Contamination profile has {contamination_profiles.shape[0]} features, "
                            f"but simulated samples have {n_features}")
        contamination_sources = [contamination_profiles]
        n_sources = 1
    
    # Set default parameters if not provided
    if contamination_params is None:
        contamination_params = {}
    
    # Handle different contamination methods
    if contamination_method == 'fixed':
        # Apply same contamination proportion to all samples
        for i in range(n_samples):
            # Randomly select a contamination source if multiple are provided
            source_idx = np.random.randint(n_sources)
            source = contamination_sources[source_idx]
            
            # Randomly select a sample from the source if it has multiple samples
            if source.shape[1] > 1:
                sample_idx = np.random.randint(source.shape[1])
                contaminant = source[:, sample_idx:sample_idx+1]
            else:
                contaminant = source
            
            # Apply fixed contamination
            prop = contamination_proportion[i]
            contaminated_samples[:, i:i+1] = (1 - prop) * contaminated_samples[:, i:i+1] + prop * contaminant
            contamination_levels[i] = prop
            
    elif contamination_method == 'random':
        # Apply random contamination proportion to each sample
        for i in range(n_samples):
            # Randomly select a contamination source if multiple are provided
            source_idx = np.random.randint(n_sources)
            source = contamination_sources[source_idx]
            
            # Randomly select a sample from the source if it has multiple samples
            if source.shape[1] > 1:
                sample_idx = np.random.randint(source.shape[1])
                contaminant = source[:, sample_idx:sample_idx+1]
            else:
                contaminant = source
            
            # Apply random contamination (up to the specified proportion)
            max_prop = contamination_proportion[i]
            prop = np.random.uniform(0, max_prop)
            contaminated_samples[:, i:i+1] = (1 - prop) * contaminated_samples[:, i:i+1] + prop * contaminant
            contamination_levels[i] = prop
            
    elif contamination_method == 'random_weighted':
        # Apply weighted average contamination with multiple sources
        weights = contamination_params.get('weights', None)
        
        if weights is None:
            # If no weights provided, use equal weights
            weights = np.ones(n_sources) / n_sources
        elif len(weights) != n_sources:
            raise ValueError(f"Number of weights ({len(weights)}) must match number of contamination sources ({n_sources})")
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        for i in range(n_samples):
            # Generate random source-specific contamination levels
            source_props = np.random.dirichlet(weights * 10)  # Alpha controls variability
            
            # Overall contamination level for this sample
            prop = contamination_proportion[i]
            
            # Start with the pure sample
            mixed_sample = (1 - prop) * contaminated_samples[:, i:i+1]
            
            # Add weighted contributions from each source
            for j, source in enumerate(contamination_sources):
                # Randomly select a sample from the source if it has multiple samples
                if source.shape[1] > 1:
                    sample_idx = np.random.randint(source.shape[1])
                    contaminant = source[:, sample_idx:sample_idx+1]
                else:
                    contaminant = source
                
                # Add this source's contribution
                source_prop = prop * source_props[j]
                mixed_sample += source_prop * contaminant
            
            contaminated_samples[:, i:i+1] = mixed_sample
            contamination_levels[i] = prop
            
    elif contamination_method == 'beta_distribution':
        # Sample contamination proportions from a beta distribution
        alpha = contamination_params.get('alpha', 2.0)
        beta = contamination_params.get('beta', 5.0)
        
        for i in range(n_samples):
            # Randomly select a contamination source if multiple are provided
            source_idx = np.random.randint(n_sources)
            source = contamination_sources[source_idx]
            
            # Randomly select a sample from the source if it has multiple samples
            if source.shape[1] > 1:
                sample_idx = np.random.randint(source.shape[1])
                contaminant = source[:, sample_idx:sample_idx+1]
            else:
                contaminant = source
            
            # Sample contamination proportion from beta distribution
            max_prop = contamination_proportion[i]
            prop = np.random.beta(alpha, beta) * max_prop
            contaminated_samples[:, i:i+1] = (1 - prop) * contaminated_samples[:, i:i+1] + prop * contaminant
            contamination_levels[i] = prop
    
    # Ensure valid methylation beta values
    contaminated_samples = np.clip(contaminated_samples, 0, 1)
    
    return contaminated_samples, contamination_levels


def save_methylation_vae_model(model, file_path, metadata=None):
    """
    Save a trained BetaVAE model for methylation data along with relevant metadata.
    
    Parameters:
    -----------
    model : BetaVAE
        Trained VAE model to save
    file_path : str
        Path where the model should be saved
    metadata : dict, optional
        Additional metadata to save with the model (e.g., feature names, 
        model parameters, training information)
    """
    # Prepare data to save
    save_data = {
        'model_state_dict': model.state_dict(),
        'input_dim': model.input_dim,
        'latent_dim': model.latent_dim,
        'hidden_dim': model.hidden_dim,
        'metadata': metadata if metadata is not None else {}
    }
    
    # Save the model and metadata
    torch.save(save_data, file_path)
    print(f"Model successfully saved to {file_path}")

def load_methylation_vae_model(file_path, device=None):
    """
    Load a trained BetaVAE model for methylation data.
    
    Parameters:
    -----------
    file_path : str
        Path to the saved model file
    device : torch.device, optional
        Device to load the model onto (cuda or cpu)
        
    Returns:
    --------
    model : BetaVAE
        Loaded VAE model
    metadata : dict
        Metadata that was saved with the model
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load saved data
    save_data = torch.load(file_path, map_location=device)
    
    # Create model with the same architecture
    model = BetaVAE(
        input_dim=save_data['input_dim'],
        latent_dim=save_data['latent_dim'],
        hidden_dim=save_data['hidden_dim']
    ).to(device)
    
    # Load the model weights
    model.load_state_dict(save_data['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    print(f"Model successfully loaded from {file_path}")
    return model, save_data['metadata']


def generate_samples_per_class_with_contamination(df, classes, samples_per_class=10, latent_dim=10, 
                                                hidden_dim=128, epochs=50, sparsity_level=0.0, 
                                                sparsity_pattern=None, contamination_labels=None, 
                                                contamination_proportion=0.0, contamination_method='random_weighted', 
                                                contamination_params=None, device=None):
    """
    Generate samples for each class with controlled sparsity and contamination.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with features as rows and samples as columns
    classes : array-like
        Array of class labels for each sample
    samples_per_class : int
        Number of samples to generate for each class
    latent_dim : int
        Dimension of the latent space
    hidden_dim : int
        Dimension of the hidden layers
    epochs : int
        Number of training epochs
    sparsity_level : float
        Proportion of values to set to NaN (0.0 to 1.0)
    sparsity_pattern : str, optional
        Type of sparsity pattern to apply
    contamination_labels : list, optional
        Labels of classes to use as contamination sources
    contamination_proportion : float
        Overall proportion of contamination to apply (0.0 to 1.0)
    contamination_method : str
        Method to apply contamination
    contamination_params : dict, optional
        Additional parameters for the contamination method
    device : torch.device
        Device to run the model on
        
    Returns:
    --------
    simulated_df : pandas.DataFrame
        Combined DataFrame with all simulated samples and metadata
        - Index: Feature names
        - Columns: Sample IDs
    metadata_df : pandas.DataFrame
        DataFrame with metadata about each sample
        - Index: Sample IDs
        - Columns: Class, contamination_level, contamination_sources
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Train the VAE
    model, col_means, device = train_beta_vae(
        df, classes, latent_dim=latent_dim, hidden_dim=hidden_dim, 
        epochs=epochs, device=device
    )

    # Save the model
    metadata = {
        'feature_names': df.index.tolist(),
        'col_means': col_means.tolist(),
        'n_classes': len(np.unique(classes)),
        'training_samples': len(classes),
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'description': 'Methylation VAE model for synthetic beta-distributed data'
    }
    save_methylation_vae_model(model, "methylation_vae_model.pth", metadata=metadata)
    
    # Get unique classes
    unique_classes = np.unique(classes)
    
    # Transpose df to get samples as rows
    X = df.T.values
    
    # Dictionary to store generated samples for each class
    simulated_data = {}
    contamination_data = {}
    
    # Generate pure samples for each class
    for class_label in unique_classes:
        # Generate pure samples
        pure_samples = simulate_methylation_samples_by_class(
            model, class_label, X, classes, 
            n_samples=samples_per_class, device=device,
            sparsity_level=0.0  # No sparsity yet, we'll apply it after contamination
        )
        
        # Store pure samples for potential use as contamination sources
        simulated_data[class_label] = pure_samples
    
    # Create contamination sources if requested
    if contamination_labels is not None and contamination_proportion > 0:
        contamination_sources = {}
        
        # Check if contamination labels are valid
        for label in contamination_labels:
            if label not in unique_classes:
                raise ValueError(f"Contamination label {label} not found in classes")
            
            # Use the pure simulated samples as contamination sources
            contamination_sources[label] = simulated_data[label]
        
        # Apply contamination to each class
        for class_label in unique_classes:
            # Skip if this class is used only as a contamination source
            if class_label in contamination_labels and class_label not in simulated_data:
                continue
                
            # Apply contamination
            contaminated_samples, contamination_levels = apply_contamination(
                simulated_data[class_label],
                contamination_sources,
                contamination_proportion,
                contamination_method,
                contamination_params
            )
            
            # Store contamination levels
            contamination_data[class_label] = {
                'levels': contamination_levels,
                'sources': contamination_labels
            }
            
            # Apply sparsity after contamination
            if sparsity_level > 0:
                # Get original data for this class to match sparsity pattern if needed
                if sparsity_pattern == 'match_original':
                    class_indices = np.where(np.array(classes) == class_label)[0]
                    original_class_data = X[class_indices].T  # Transpose to features as rows
                else:
                    original_class_data = None
                
                contaminated_samples = apply_sparsity(
                    contaminated_samples, 
                    sparsity_level, 
                    sparsity_pattern, 
                    original_class_data
                )
            
            # Update the simulated data with contaminated samples
            simulated_data[class_label] = contaminated_samples
    else:
        # Apply sparsity to pure samples if no contamination
        for class_label in unique_classes:
            if sparsity_level > 0:
                # Get original data for this class to match sparsity pattern if needed
                if sparsity_pattern == 'match_original':
                    class_indices = np.where(np.array(classes) == class_label)[0]
                    original_class_data = X[class_indices].T  # Transpose to features as rows
                else:
                    original_class_data = None
                
                simulated_data[class_label] = apply_sparsity(
                    simulated_data[class_label], 
                    sparsity_level, 
                    sparsity_pattern, 
                    original_class_data
                )
            
            # Add empty contamination data for consistency
            contamination_data[class_label] = {
                'levels': np.zeros(simulated_data[class_label].shape[1]),
                'sources': []
            }
    
    # Prepare data for the combined DataFrame
    feature_names = df.index.tolist()
    
    # Combine all simulated samples into a single matrix
    all_samples_list = []
    sample_metadata = {
        'sample_id': [],
        'class': [],
        'contamination_level': [],
        'contamination_sources': []
    }
    
    for class_label, samples in simulated_data.items():
        # Add the samples
        all_samples_list.append(samples)
        
        # Add metadata for each sample
        n_samples = samples.shape[1]
        sample_metadata['sample_id'].extend([f"sim_{class_label}_{i}" for i in range(n_samples)])
        sample_metadata['class'].extend([class_label] * n_samples)
        
        # Add contamination information
        if class_label in contamination_data:
            sample_metadata['contamination_level'].extend(contamination_data[class_label]['levels'])
            sample_metadata['contamination_sources'].extend([contamination_data[class_label]['sources']] * n_samples)
        else:
            sample_metadata['contamination_level'].extend([0.0] * n_samples)
            sample_metadata['contamination_sources'].extend([None] * n_samples)
    
    # Create the combined data matrix
    if all_samples_list:
        all_samples_matrix = np.hstack(all_samples_list)
        
        # Create the main DataFrame with methylation values
        simulated_df = pd.DataFrame(
            all_samples_matrix,
            index=feature_names,
            columns=sample_metadata['sample_id']
        )
        
        # Create a metadata DataFrame
        metadata_df = pd.DataFrame({
            'class': sample_metadata['class'],
            'contamination_level': sample_metadata['contamination_level'],
            'contamination_sources': sample_metadata['contamination_sources']
        }, index=sample_metadata['sample_id'])
        
        return simulated_df, metadata_df
    else:
        return pd.DataFrame(), pd.DataFrame()


# Example usage
betas = pd.read_parquet("HM450_reference_v3.parquet")
m = pd.read_parquet("reference_v3.parquet")
classes = m.loc[:,"entity"].values
feature_names = betas.index.values
simulated_df, metadata_df = generate_samples_per_class_with_contamination(
    betas, classes, samples_per_class=10, latent_dim=10, hidden_dim=128, 
    epochs=50, sparsity_level=0.0, sparsity_pattern=None, 
    contamination_labels=None, contamination_proportion=0.0, 
    contamination_method='random_weighted', contamination_params=None
)
simulated_df.to_parquet("simulated_methylation_samples.parquet")
metadata_df.to_parquet("simulated_metadata.parquet")
print("Simulated methylation samples and metadata saved.")