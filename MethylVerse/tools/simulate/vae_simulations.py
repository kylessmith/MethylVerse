import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from tqdm import tqdm

class ConditionalBetaVAE(nn.Module):
    """
    Conditional Variational Autoencoder optimized for beta-distributed methylation data.
    Simplified 2-layer architecture.
    """
    def __init__(self, input_dim, num_classes, latent_dim=20, hidden_dim=256, dropout_rate=0.3):
        super(ConditionalBetaVAE, self).__init__()
        
        # Simplified encoder with 2 layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        
        # Latent space projections
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Class embedding for conditioning
        self.class_embedding = nn.Embedding(num_classes, latent_dim)
        
        # Simplified decoder with 2 layers
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_hidden = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        self.decoder_output = nn.Linear(hidden_dim, input_dim)
        
        # Add variational mixture of posteriors prior
        self.vamp_pseudoinputs = nn.Parameter(torch.randn(50, input_dim))
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
    
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
    
    def decode(self, z, class_indices=None):
        """
        Decode latent space representation to output space.
        Incorporates class information if provided.
        """
        # Apply class conditioning if provided
        if class_indices is not None:
            class_embed = self.class_embedding(class_indices)
            z = z + class_embed
            
        z = self.decoder_input(z)
        z = self.decoder_hidden(z)
        x_logits = self.decoder_output(z)
        
        # Custom bounded sigmoid to avoid exact 0 and 1
        # Sigmoid normally maps to (0,1), this maps to (0.001, 0.999)
        x_recon = 0.998 * torch.sigmoid(x_logits) + 0.001
        
        return x_recon
    
    def forward(self, x, class_indices=None):
        """
        Forward pass through the VAE.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, class_indices)
        return x_recon, mu, logvar
    
    def generate(self, z, class_indices=None):
        """
        Generate samples from latent vectors.
        """
        return self.decode(z, class_indices)
    
    def get_vamp_prior(self, n_samples=None):
        """
        Get samples from the VampPrior (variational mixture of posteriors).
        This helps create a more expressive prior distribution.
        """
        if n_samples is None:
            n_samples = self.vamp_pseudoinputs.size(0)
        else:
            # Random selection of pseudoinputs
            idx = torch.randint(0, self.vamp_pseudoinputs.size(0), (n_samples,))
            pseudoinputs = self.vamp_pseudoinputs[idx]
            
        # Ensure pseudoinputs are in proper range for methylation data
        pseudoinputs_activated = torch.sigmoid(self.vamp_pseudoinputs)
        
        # Encode the pseudoinputs
        mu, logvar = self.encode(pseudoinputs_activated)
        
        # Sample from the encoded distribution
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar  # Return mu and logvar as well for direct KL calculation
    

    def extract_latent_space(self, x, return_distribution=True, samples=1):
        """
        Extract latent space representation from input data.
        
        Args:
            x (torch.Tensor): Input data tensor of shape [batch_size, input_dim]
            return_distribution (bool): If True, returns mu and logvar along with samples
            samples (int): Number of samples to draw from the latent distribution
            
        Returns:
            If return_distribution=True:
                (z, mu, logvar) where:
                    - z is the sampled latent vectors [samples, batch_size, latent_dim]
                    - mu is the mean of the latent distribution [batch_size, latent_dim]
                    - logvar is the log variance of the latent distribution [batch_size, latent_dim]
            If return_distribution=False:
                z: Just the sampled latent vectors [samples, batch_size, latent_dim]
        """
        # Ensure evaluation mode for consistent results
        self.eval()
        
        # Handle NaN inputs as in the encode method
        if torch.isnan(x).any():
            print("Warning: NaN detected in input data")
            x = torch.nan_to_num(x, nan=0.5)
        
        # Get latent distribution parameters
        with torch.no_grad():
            mu, logvar = self.encode(x)
            
            # Draw multiple samples if requested
            if samples > 1:
                batch_size = x.size(0)
                
                # Expand mu and logvar for multiple samples
                mu_expanded = mu.unsqueeze(0).expand(samples, -1, -1)  # [samples, batch, latent_dim]
                logvar_expanded = logvar.unsqueeze(0).expand(samples, -1, -1)
                
                # Sample from the expanded distribution
                std = torch.exp(0.5 * logvar_expanded)
                eps = torch.randn_like(std)
                z = mu_expanded + eps * std
            else:
                # Single sample (standard case)
                z = self.reparameterize(mu, logvar)
                z = z.unsqueeze(0)  # Add samples dimension for consistency
        
        # Return results
        if return_distribution:
            return z, mu, logvar
        else:
            return z
        
    
    def latent_to_features(self, z, class_indices=None, return_logits=False):
        """
        Convert latent space representations back to the original feature space.
        
        Args:
            z (torch.Tensor): Latent vectors with shape [batch_size, latent_dim] or 
                            [samples, batch_size, latent_dim] for multiple samples
            class_indices (torch.Tensor, optional): Class indices for conditional generation
                                                Shape should be [batch_size] or [samples, batch_size]
            return_logits (bool): If True, returns raw logits before sigmoid activation
            
        Returns:
            torch.Tensor: Reconstructed features in the original feature space
                        Shape will be [batch_size, input_dim] or [samples, batch_size, input_dim]
        """
        # Ensure evaluation mode for consistent results
        self.eval()
        
        # Handle case where z has a samples dimension
        original_shape = z.shape
        multi_sample = len(original_shape) == 3
        
        with torch.no_grad():
            if multi_sample:
                samples, batch_size, latent_dim = z.shape
                
                # Reshape for processing
                z_flat = z.reshape(-1, latent_dim)  # [samples*batch, latent_dim]
                
                # Handle class indices for multi-sample case
                if class_indices is not None:
                    if len(class_indices.shape) == 1:  # [batch]
                        # Expand class indices to match samples
                        class_indices = class_indices.unsqueeze(0).expand(samples, -1)  # [samples, batch]
                    
                    # Flatten class indices to match z_flat
                    class_indices_flat = class_indices.reshape(-1)  # [samples*batch]
                else:
                    class_indices_flat = None
                
                # Decode flattened latent vectors
                if return_logits:
                    # Custom implementation to get logits before sigmoid
                    # Apply class conditioning if provided
                    if class_indices_flat is not None:
                        class_embed = self.class_embedding(class_indices_flat)
                        z_cond = z_flat + class_embed
                    else:
                        z_cond = z_flat
                        
                    z_cond = self.decoder_input(z_cond)
                    z_cond = self.decoder_hidden(z_cond)
                    logits = self.decoder_output(z_cond)
                    
                    # Reshape back to [samples, batch, input_dim]
                    reconstructed = logits.reshape(samples, batch_size, self.input_dim)
                else:
                    # Use the existing decode method
                    reconstructed = self.decode(z_flat, class_indices_flat)
                    
                    # Reshape back to [samples, batch, input_dim]
                    reconstructed = reconstructed.reshape(samples, batch_size, self.input_dim)
            else:
                # Single sample dimension case - use standard decoding
                if return_logits:
                    # Custom implementation to get logits before sigmoid
                    # Apply class conditioning if provided
                    if class_indices is not None:
                        class_embed = self.class_embedding(class_indices)
                        z_cond = z + class_embed
                    else:
                        z_cond = z
                        
                    z_cond = self.decoder_input(z_cond)
                    z_cond = self.decoder_hidden(z_cond)
                    reconstructed = self.decoder_output(z_cond)
                else:
                    # Use the existing decode method
                    reconstructed = self.decode(z, class_indices)
        
        return reconstructed


def advanced_beta_loss_function(recon_x, x, mu, logvar, model=None, kl_weight=0.5, 
                              use_vamp_prior=True, consistency_weight=0.2):
    """
    Enhanced loss function for beta-distributed data with VampPrior and consistency regularization.
    
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
    model : ConditionalBetaVAE
        Model instance for VampPrior calculation
    kl_weight : float
        Weight for the KL divergence term
    use_vamp_prior : bool
        Whether to use VampPrior instead of standard normal prior
    consistency_weight : float
        Weight for the consistency regularization term
        
    Returns:
    --------
    total_loss : torch.Tensor
        Combined loss (reconstruction + KL divergence + consistency)
    recon_loss : torch.Tensor
        Reconstruction loss component
    kl_loss : torch.Tensor
        KL divergence loss component
    
    Notes:
    ------
    This function handles potential batch size mismatches that can occur when using VampPrior.
    """
    # Use beta distribution loss for methylation data (0-1 range)
    # This better models the bimodal nature of methylation values
    eps = 1e-6  # Small epsilon to prevent log(0)
    x_clamped = torch.clamp(x, eps, 1-eps)
    recon_x_clamped = torch.clamp(recon_x, eps, 1-eps)
    
    # Beta distribution-based loss for methylation data
    # We use a weighted combination of MSE and beta log-likelihood
    mse_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # Beta log-likelihood term (simplified version)
    beta_loss = -torch.mean(
        x_clamped * torch.log(recon_x_clamped) + 
        (1 - x_clamped) * torch.log(1 - recon_x_clamped)
    )
    
    # Combine the losses with appropriate weighting
    recon_loss = 0.7 * beta_loss + 0.3 * mse_loss
    
    # KL divergence with numerically stable implementation
    try:
        if use_vamp_prior and model is not None:
            # VampPrior KL divergence calculation
            try:
                if use_vamp_prior and model is not None:
                    # Get VampPrior parameters directly (avoids need to decode then re-encode)
                    _, vamp_mu, vamp_logvar = model.get_vamp_prior(n_samples=mu.size(0))
                    
                    # Calculate KL using Monte Carlo sampling from the approximate posterior
                    # See "Variational Inference with Vampprior" (Tomczak & Welling, 2018)
                    
                    batch_size = mu.size(0)
                    z_samples = model.reparameterize(mu, logvar)  # Samples from q(z|x)
                    
                    # Calculate log q(z|x) for each sample
                    log_q_z_x = log_normal_pdf(z_samples, mu, logvar).sum(dim=1)
                    
                    # Calculate log p(z) as log-sum-exp of mixture components
                    vamp_components = vamp_mu.size(0)  # Number of pseudoinputs
                    z_samples_expanded = z_samples.unsqueeze(1).expand(-1, vamp_components, -1)  # [batch, components, latent_dim]
                    vamp_mu_expanded = vamp_mu.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, components, latent_dim]
                    vamp_logvar_expanded = vamp_logvar.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, components, latent_dim]
                    
                    # Calculate log p(z|u_k) for each component
                    log_p_z_uk = log_normal_pdf(z_samples_expanded, vamp_mu_expanded, vamp_logvar_expanded).sum(dim=2)  # [batch, components]
                    
                    # Log-sum-exp trick for numerical stability
                    max_log_p_z_uk = torch.max(log_p_z_uk, dim=1, keepdim=True)[0]
                    log_p_z = max_log_p_z_uk.squeeze() + torch.log(torch.exp(log_p_z_uk - max_log_p_z_uk).sum(dim=1))
                    
                    # KL divergence: E_q(z|x)[log q(z|x) - log p(z)]
                    kl_loss = torch.mean(log_q_z_x - log_p_z)
                else:
                    # Standard KL to normal prior
                    logvar = torch.clamp(logvar, -10.0, 10.0)  # Prevent extreme values
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            except Exception as e:
                print(f"VampPrior calculation failed: {e}. Using standard KL divergence.")
                # Standard KL to normal prior as fallback
                logvar = torch.clamp(logvar, -10.0, 10.0)  # Prevent extreme values
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            # Standard KL to normal prior
            logvar = torch.clamp(logvar, -10.0, 10.0)  # Prevent extreme values
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Handle potential NaN or inf in KL loss
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            print("Warning: NaN or Inf in KL loss, using fallback")
            kl_loss = torch.tensor(0.1, device=recon_x.device)
    except Exception as e:
        print(f"Error in KL loss calculation: {e}")
        kl_loss = torch.tensor(0.1, device=recon_x.device)
    
    # Add consistency regularization to promote smoother interpolations in latent space
    consistency_loss = 0.0
    if consistency_weight > 0 and model is not None:  # Check if model is provided
        # Generate random interpolations between samples in the batch
        batch_size = x.size(0)
        if batch_size > 1:
            alpha = torch.rand(batch_size, 1, device=x.device)
            indices = torch.randperm(batch_size, device=x.device)
            
            # Interpolate inputs
            x_interp = alpha * x + (1 - alpha) * x[indices]
            
            # Get encodings for interpolated inputs
            with torch.no_grad():
                mu_interp, logvar_interp = model.encode(x_interp)
                z_interp = model.reparameterize(mu_interp, logvar_interp)
            
            # Check if interpolated encodings are close to interpolated latent vectors
            z_target = alpha * model.reparameterize(mu, logvar) + (1 - alpha) * model.reparameterize(mu[indices], logvar[indices])
            consistency_loss = F.mse_loss(z_interp, z_target)
    else:
        # Skip consistency regularization if model not provided
        consistency_loss = torch.tensor(0.0, device=x.device)
    # Combined loss with weights
    total_loss = recon_loss + kl_weight * kl_loss + consistency_weight * consistency_loss
    
    return total_loss, recon_loss, kl_loss


def log_normal_pdf(x, mu, logvar):
    """
    Compute the log PDF of a normal distribution.
    Used for VampPrior calculations.
    
    Parameters:
    -----------
    x : torch.Tensor
        Points at which to evaluate the log PDF
    mu : torch.Tensor
        Mean of the normal distribution
    logvar : torch.Tensor
        Log variance of the normal distribution
        
    Returns:
    --------
    log_pdf : torch.Tensor
        Log PDF values
    """
    const = torch.log(torch.tensor(2 * np.pi, device=x.device))
    return -0.5 * (const + logvar + (x - mu).pow(2) / torch.exp(logvar))


def log_normal_mixture(x, mu, logvar):
    """
    Compute the log PDF of a mixture of normal distributions.
    Used for VampPrior calculations.
    """
    # Ensure consistent batch sizes
    x_batch_size = x.size(0)
    mu_batch_size = mu.size(0)
    
    # Handle case when batch sizes don't match
    if x_batch_size != mu_batch_size:
        # Option 1: Repeat the smaller tensor to match the larger one
        if x_batch_size < mu_batch_size:
            # Select a subset of mu/logvar to match x's batch size
            mu = mu[:x_batch_size]
            logvar = logvar[:x_batch_size]
        else:
            # Select a subset of x to match mu's batch size
            x = x[:mu_batch_size]
    
    x = x.unsqueeze(1)  # [batch_size, 1, latent_dim]
    mu = mu.unsqueeze(0)  # [1, batch_size, latent_dim]
    logvar = logvar.unsqueeze(0)  # [1, batch_size, latent_dim]
    
    log_probs = log_normal_pdf(x, mu, logvar)
    log_probs = log_probs.sum(dim=-1)  # Sum over latent dimensions
    
    # Log-sum-exp trick for numerical stability
    max_log_probs = torch.max(log_probs, dim=1, keepdim=True)[0]
    return max_log_probs + torch.log(torch.exp(log_probs - max_log_probs).sum(dim=1))


def augment_batch(batch, noise_level=0.05, p_flip=0.02):
    """
    Augment a batch of data with realistic methylation-specific transformations.
    
    Parameters:
    -----------
    batch : torch.Tensor
        Batch of methylation data
    noise_level : float
        Level of Gaussian noise to add
    p_flip : float
        Probability of flipping methylation status (fully methylated to unmethylated or vice versa)
        
    Returns:
    --------
    augmented_batch : torch.Tensor
        Augmented batch
    """
    batch_shape = batch.shape
    augmented = batch.clone()
    
    # Add small random noise to methylation values
    noise = torch.randn_like(augmented) * noise_level
    augmented = augmented + noise
    
    # Randomly flip some methylation sites (0->1 or 1->0)
    flip_mask = torch.bernoulli(torch.ones_like(augmented) * p_flip).bool()
    augmented[flip_mask] = 1.0 - augmented[flip_mask]
    
    # Make CpG site-specific adjustments
    # Some sites tend to be more unstable in methylation status
    site_noise = torch.rand(batch_shape[1], device=batch.device) * 0.1
    site_noise = site_noise.unsqueeze(0).expand(batch_shape)
    augmented = augmented + (torch.randn_like(augmented) * site_noise)
    
    # Ensure values stay in valid range
    augmented = torch.clamp(augmented, 1e-6, 1 - 1e-6)
    
    return augmented


def train_advanced_vae(df, classes, test_size=0.2, latent_dim=20, hidden_dim=256, 
                      epochs=100, batch_size=32, device=None, learning_rate=1e-4,
                      gradient_clip=1.0, early_stop_patience=10, beta_weight_start=0.0,
                      beta_weight_end=0.5, beta_warmup_epochs=20, use_vamp_prior=True,
                      consistency_weight=0.2, augmentation_strength=0.05):
    """
    Train an advanced Conditional VAE on methylation data with improved stability and diversity.
    Includes VampPrior, consistency regularization, and data augmentation.
    
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
    use_vamp_prior : bool
        Whether to use VampPrior instead of standard normal prior
    consistency_weight : float
        Weight for the consistency regularization term
    augmentation_strength : float
        Strength of data augmentation during training
        
    Returns:
    --------
    model : ConditionalBetaVAE
        Trained VAE model
    col_means : numpy.ndarray
        Column means used for imputation
    class_mapping : dict
        Mapping from class labels to numeric indices
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
        X = np.clip(X, 0, 1)
    
    # Create class mapping for model
    unique_classes = np.unique(classes)
    class_mapping = {c: i for i, c in enumerate(unique_classes)}
    class_indices = np.array([class_mapping[c] for c in classes])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, class_indices, test_size=test_size, stratify=class_indices, random_state=42
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
    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create and initialize the conditional model
    input_dim = X_train.shape[1]
    num_classes = len(unique_classes)
    model = ConditionalBetaVAE(input_dim, num_classes, latent_dim, hidden_dim).to(device)
    
    # Use more reliable Xavier uniform initialization with small gain adjustment
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.8)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # Use AdamW optimizer with weight decay for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Cosine annealing learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate / 10
    )
    
    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training history
    history = {
        'train_loss': [],
        'test_loss': [],
        'recon_loss': [],
        'kl_loss': [],
        'beta': []
    }
    
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
            # Smoother warmup with sigmoid function
            progress = epoch / beta_warmup_epochs
            beta = beta_weight_start + (beta_weight_end - beta_weight_start) * (
                1 / (1 + np.exp(-12 * (progress - 0.5)))
            )
        else:
            beta = beta_weight_end
        
        history['beta'].append(beta)
        
        for batch_idx, (data, class_idx) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            # Check for NaN in input data
            if torch.isnan(data).any():
                print(f"Warning: NaN found in input batch {batch_idx}, replacing with 0.5")
                data = torch.nan_to_num(data, nan=0.5)
                
            # Check for extreme values
            if data.min() < 1e-6 or data.max() > 1 - 1e-6:
                data = torch.clamp(data, 1e-6, 1 - 1e-6)
            
            # Apply data augmentation
            if augmentation_strength > 0:
                data = augment_batch(data, noise_level=augmentation_strength)
            
            optimizer.zero_grad()
            
            try:
                # Forward pass with class conditioning
                recon_batch, mu, logvar = model(data, class_idx)
                
                # Calculate loss with current beta value and VampPrior
                try:
                    loss, recon_loss, kl_loss = advanced_beta_loss_function(
                        recon_batch, data, mu, logvar, model=model if use_vamp_prior else None,
                        kl_weight=beta, use_vamp_prior=use_vamp_prior, 
                        consistency_weight=consistency_weight
                    )
                except Exception as e:
                    print(f"Error in loss calculation: {e}")
                    print("Falling back to standard KL loss without consistency regularization")
                    # Fallback to standard KL loss without consistency regularization
                    eps = 1e-6  # Small epsilon to prevent log(0)
                    x_clamped = torch.clamp(data, eps, 1-eps)
                    recon_x_clamped = torch.clamp(recon_batch, eps, 1-eps)
                    
                    # Beta log-likelihood term (simplified version)
                    beta_loss = -torch.mean(
                        x_clamped * torch.log(recon_x_clamped) + 
                        (1 - x_clamped) * torch.log(1 - recon_x_clamped)
                    )
                    
                    # Standard KL to normal prior
                    logvar_clamped = torch.clamp(logvar, -10.0, 10.0)
                    kl_div = -0.5 * torch.mean(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp())
                    
                    # Combine losses
                    recon_loss = beta_loss
                    kl_loss = kl_div
                    loss = recon_loss + beta * kl_loss
                
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
        
        # Update the learning rate
        scheduler.step()
        
        # Record training history
        history['train_loss'].append(avg_train_loss)
        history['recon_loss'].append(avg_train_recon)
        history['kl_loss'].append(avg_train_kl)
        
        # Evaluation mode
        model.eval()
        test_loss = 0
        test_batch_count = 0
        
        with torch.no_grad():
            for batch_idx, (data, class_idx) in enumerate(test_loader):
                try:
                    # Replace NaNs and clip extreme values
                    if torch.isnan(data).any():
                        data = torch.nan_to_num(data, nan=0.5)
                    data = torch.clamp(data, 1e-6, 1 - 1e-6)
                    
                    recon_batch, mu, logvar = model(data, class_idx)
                    loss, _, _ = advanced_beta_loss_function(
                        recon_batch, data, mu, logvar, model=model if use_vamp_prior else None,
                        kl_weight=beta, use_vamp_prior=use_vamp_prior, 
                        consistency_weight=consistency_weight
                    )
                    
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
            history['test_loss'].append(float('nan'))
            continue
            
        avg_test_loss = test_loss / test_batch_count
        history['test_loss'].append(avg_test_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}, Beta: {beta:.4f}, Train Loss: {avg_train_loss:.4f} '
              f'(Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f}), '
              f'Test Loss: {avg_test_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Early stopping check
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            patience_counter = 0
            # Save best model
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"New best model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                # Restore best model
                if best_model_state is not None:
                    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                break
    
    # Plot training history
    try:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['test_loss'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Total Loss')
        
        plt.subplot(2, 2, 2)
        plt.plot(history['recon_loss'], label='Reconstruction Loss')
        plt.plot(history['kl_loss'], label='KL Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Component')
        plt.legend()
        plt.title('Loss Components')
        
        plt.subplot(2, 2, 3)
        plt.plot(history['beta'], label='Beta Value')
        plt.xlabel('Epoch')
        plt.ylabel('Beta')
        plt.title('Beta Annealing Schedule')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history plot saved to 'training_history.png'")
    except Exception as e:
        print(f"Error creating training plot: {e}")
    
    # Final evaluation with simplified loss
    model.eval()
    test_loss = 0
    test_batch_count = 0
    
    with torch.no_grad():
        for batch_idx, (data, class_idx) in enumerate(test_loader):
            # Replace NaNs and clip extreme values
            if torch.isnan(data).any():
                data = torch.nan_to_num(data, nan=0.5)
            data = torch.clamp(data, 1e-6, 1 - 1e-6)
            
            recon_batch, mu, logvar = model(data, class_idx)
            
            # Use simple loss calculation for final evaluation
            recon_loss = F.mse_loss(recon_batch, data)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta_weight_end * kl_loss
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                test_loss += loss.item()
                test_batch_count += 1
    
    if test_batch_count > 0:
        print(f'Final Test Loss: {test_loss / test_batch_count:.4f}')
    else:
        print("Warning: Could not compute final test loss due to numerical issues")
    
    return model, col_means, class_mapping, device


def save_methylation_vae_model(model, file_path, metadata=None):
    """
    Save a trained Conditional BetaVAE model for methylation data along with relevant metadata.
    
    Parameters:
    -----------
    model : ConditionalBetaVAE
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
        'num_classes': model.num_classes,
        'metadata': metadata if metadata is not None else {}
    }
    
    # Save the model and metadata
    torch.save(save_data, file_path)
    print(f"Model successfully saved to {file_path}")


def load_methylation_vae_model(file_path, device=None):
    """
    Load a trained Conditional BetaVAE model for methylation data.
    
    Parameters:
    -----------
    file_path : str
        Path to the saved model file
    device : torch.device, optional
        Device to load the model onto (cuda or cpu)
        
    Returns:
    --------
    model : ConditionalBetaVAE
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
    model = ConditionalBetaVAE(
        input_dim=save_data['input_dim'],
        num_classes=save_data['num_classes'],
        latent_dim=save_data['latent_dim'],
        hidden_dim=save_data['hidden_dim']
    ).to(device)
    
    # Load the model weights
    model.load_state_dict(save_data['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    print(f"Model successfully loaded from {file_path}")
    return model, save_data['metadata']


def train_vae_model(df,
                    classes, 
                    latent_dim=20, 
                    hidden_dim=256,
                    epochs=100,
                    augmentation_strength=0.05,
                    beta_weight_end=0.5,
                    consistency_weight=0.2,
                    batch_size=32,
                    device=None,
                    save_model="advanced_methylation_vae_model.pth",
                    use_vamp_prior=False):
    """
    Generate samples for each class with controlled sparsity and contamination.
    Enhanced version with improved diversity, more realistic sparsity patterns,
    and comprehensive evaluation metrics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with features as rows and samples as columns
    classes : array-like
        Array of class labels for each sample
    latent_dim : int
        Dimension of the latent space
    hidden_dim : int
        Dimension of the hidden layers
    epochs : int
        Number of training epochs
    augmentation_strength : float
        Strength of data augmentation during training
    beta_weight_end : float
        Final weight for KL divergence term
    consistency_weight : float
        Weight for the consistency regularization term
    device : torch.device
        Device to run the model on
    save_model : str
        Path to save the trained model
    use_vamp_prior : bool
        Whether to use VampPrior instead of standard normal prior
        
    Returns:
    --------
    simulated_df : pandas.DataFrame
        Combined DataFrame with all simulated samples and metadata
        - Index: Feature names
        - Columns: Sample IDs
    metadata_df : pandas.DataFrame
        DataFrame with metadata about each sample
        - Index: Sample IDs
        - Columns: Class, contamination_level, contamination_sources, diversity_metrics
    model : ConditionalBetaVAE
        Trained VAE model
    diversity_metrics : dict
        Dictionary with diversity metrics per class
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Create class mappings
    unique_classes = np.unique(classes)
    class_mapping = {c: i for i, c in enumerate(unique_classes)}
    
    # Print configuration
    print("=== Configuration ===")
    print(f"Number of features: {df.shape[0]}")
    print(f"Number of samples: {df.shape[1]}")
    print(f"Number of classes: {len(unique_classes)}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Training epochs: {epochs}")
    print(f"Device: {device}")
    print("====================")
        
    # Train the advanced VAE
    # Let's also update beta_weight_end to 1.0 to encourage more diverse samples
    model, col_means, class_mapping, device = train_advanced_vae(
        df, classes, latent_dim=latent_dim, hidden_dim=hidden_dim, 
        epochs=epochs, device=device, beta_weight_end=beta_weight_end,
        use_vamp_prior=use_vamp_prior, consistency_weight=consistency_weight,
        augmentation_strength=augmentation_strength, batch_size=batch_size
    )

    # Save the model if requested
    metadata = {
        'feature_names': df.index.tolist(),
        'class_mapping': class_mapping,
        'inverse_class_mapping': {str(v): k for k, v in class_mapping.items()},
        'n_classes': len(unique_classes),
        'training_samples': len(classes),
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'description': 'Advanced Conditional VAE model for diverse methylation simulation'
    }
    save_methylation_vae_model(model, save_model, metadata=metadata)


def augment_with_latent_noise(model, samples, class_indices=None, noise_scale=0.1, 
                             noise_type='gaussian', n_augmentations=1, 
                             dimensions=None, return_latents=False):
    """
    Take samples, extract latent representations, add meaningful noise,
    and decode back to the original feature space.
    
    Args:
        model (ConditionalBetaVAE): The trained VAE model
        samples (torch.Tensor): Input samples of shape [batch_size, input_dim]
        class_indices (torch.Tensor, optional): Class indices for conditional generation
                                              Shape should be [batch_size]
        noise_scale (float): Scale of noise to add (0.1 = 10% of latent std)
        noise_type (str): Type of noise to add:
                         'gaussian' - Random normal noise
                         'targeted' - Only add noise to specific dimensions
                         'interpolate' - Interpolate between samples in latent space
                         'uniform' - Uniform random noise
        n_augmentations (int): Number of augmented samples to generate per input
        dimensions (list, optional): Specific latent dimensions to perturb (for 'targeted')
        return_latents (bool): If True, also return the original and noisy latent vectors
    
    Returns:
        augmented_samples (torch.Tensor): Augmented samples in original feature space
                                        Shape [n_augmentations, batch_size, input_dim]
        (Optional if return_latents=True):
        z (torch.Tensor): Original latent vectors [1, batch_size, latent_dim]
        z_noisy (torch.Tensor): Noisy latent vectors [n_augmentations, batch_size, latent_dim]
    """
    # Set model to eval mode
    model.eval()
    
    # Extract latent representations
    with torch.no_grad():
        # Get latent vectors and distribution parameters
        z, mu, logvar = model.extract_latent_space(samples, return_distribution=True, samples=1)
        
        # Calculate the standard deviation of each dimension
        std = torch.exp(0.5 * logvar)  # [batch_size, latent_dim]
        
        # Create noisy versions of the latent vectors
        batch_size = samples.size(0)
        latent_dim = z.size(2)
        
        # Initialize tensor to hold noisy latent vectors
        z_noisy = torch.zeros(n_augmentations, batch_size, latent_dim, device=z.device)
        
        # Generate the noise based on the specified type
        if noise_type == 'gaussian':
            # Standard Gaussian noise scaled by the latent std
            for i in range(n_augmentations):
                # Scale noise by the standard deviation of each dimension
                noise = torch.randn_like(z[0]) * std * noise_scale
                z_noisy[i] = z[0] + noise
                
        elif noise_type == 'targeted':
            # Only add noise to specified dimensions
            if dimensions is None:
                # If no dimensions specified, randomly select half
                n_dims = latent_dim // 2
                dimensions = torch.randperm(latent_dim)[:n_dims].tolist()
            
            # Create noise mask (1 for dimensions to perturb, 0 for others)
            mask = torch.zeros(latent_dim, device=z.device)
            mask[dimensions] = 1.0
            
            for i in range(n_augmentations):
                # Generate noise for all dimensions
                noise = torch.randn_like(z[0]) * std * noise_scale
                # Apply mask to only affect target dimensions
                masked_noise = noise * mask
                z_noisy[i] = z[0] + masked_noise
                
        elif noise_type == 'interpolate':
            # Interpolate between samples in the batch
            if batch_size < 2:
                raise ValueError("Need at least 2 samples for interpolation")
            
            for i in range(n_augmentations):
                # For each augmentation, create random pairs of indices to interpolate
                idx1 = torch.randint(0, batch_size, (batch_size,))
                idx2 = torch.randint(0, batch_size, (batch_size,))
                
                # Random interpolation weight for each sample
                alpha = torch.rand(batch_size, 1, device=z.device)
                
                # Interpolate
                z1 = z[0][idx1]
                z2 = z[0][idx2]
                z_noisy[i] = alpha * z1 + (1 - alpha) * z2
                
        elif noise_type == 'uniform':
            # Uniform noise in [-noise_scale, noise_scale] * std
            for i in range(n_augmentations):
                noise = (torch.rand_like(z[0]) * 2 - 1) * std * noise_scale
                z_noisy[i] = z[0] + noise
                
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Decode the noisy latent vectors back to feature space
        augmented_samples = model.latent_to_features(z_noisy, class_indices=class_indices)
    
    if return_latents:
        return augmented_samples, z, z_noisy
    else:
        return augmented_samples