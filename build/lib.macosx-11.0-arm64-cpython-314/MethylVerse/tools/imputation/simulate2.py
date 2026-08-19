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
    Incorporates class information to better capture class-specific variations.
    """
    def __init__(self, input_dim, num_classes, latent_dim=20, hidden_dim=256, dropout_rate=0.3):
        super(ConditionalBetaVAE, self).__init__()
        
        # Enhanced encoder with LeakyReLU and deeper architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        
        # Latent space projections
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Class embedding for conditioning
        self.class_embedding = nn.Embedding(num_classes, latent_dim)
        
        # Enhanced decoder with LeakyReLU
        self.decoder_input = nn.Linear(latent_dim, hidden_dim // 2)
        self.decoder_hidden = nn.Sequential(
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
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


def apply_sparsity(data, sparsity_level=0.1, sparsity_pattern=None, original_data=None):
    """
    Apply sparsity to the data by randomly setting values to NaN with improved biological realism.
    
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
        - 'biologically_inspired': Apply biologically plausible missing patterns
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
        # Use beta distribution with shape parameters to model real-world behavior
        feature_missing_prob = np.random.beta(2, 5, size=n_features)
        feature_missing_prob = feature_missing_prob / np.sum(feature_missing_prob) * n_nans
        
        for i in range(n_features):
            n_missing = int(feature_missing_prob[i])
            if n_missing > 0:
                missing_cols = np.random.choice(n_samples, min(n_missing, n_samples), replace=False)
                sparse_data[i, missing_cols] = np.nan
                
    elif sparsity_pattern == 'sample_based':
        # Some samples are more likely to have missing values
        # E.g., poor quality DNA samples tend to have more missing values overall
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
    
    elif sparsity_pattern == 'biologically_inspired':
        # Apply biologically plausible missing patterns
        
        # 1. GC content-based missingness (CpG sites with high GC content tend to have better coverage)
        # Simulate GC content with beta distribution skewed towards higher values
        gc_content = np.random.beta(5, 2, size=n_features)
        
        # 2. Spatial correlation (adjacent CpG sites often have correlated missingness)
        # Simulate spatial correlation by applying a smoothing filter
        spatial_corr = np.zeros(n_features)
        window_size = 5
        for i in range(n_features):
            start = max(0, i - window_size // 2)
            end = min(n_features, i + window_size // 2 + 1)
            spatial_corr[i] = np.mean(gc_content[start:end])
        
        # 3. Sample quality effects (some samples have poor overall coverage)
        sample_quality = np.random.beta(4, 2, size=n_samples)
        
        # Combine these factors into a probability matrix
        prob_matrix = np.zeros((n_features, n_samples))
        for j in range(n_samples):
            # Combine GC content effect with sample quality
            prob_matrix[:, j] = (1 - spatial_corr) * (1 - sample_quality[j])
        
        # Normalize to get desired number of NaNs
        prob_matrix = prob_matrix / np.sum(prob_matrix) * n_nans
        
        # Apply probabilistic missingness
        for i in range(n_features):
            for j in range(n_samples):
                if np.random.rand() < prob_matrix[i, j] / total_elements * n_nans:
                    sparse_data[i, j] = np.nan
    
    # Ensure we have exactly the right number of NaNs (if possible)
    actual_nans = np.isnan(sparse_data).sum()
    
    # If we have too few NaNs, add more randomly
    if actual_nans < n_nans:
        # Find positions that aren't already NaN
        non_nan_mask = ~np.isnan(sparse_data)
        non_nan_positions = np.where(non_nan_mask.flatten())[0]
        
        # How many more NaNs do we need?
        n_more_nans = n_nans - actual_nans
        
        if len(non_nan_positions) >= n_more_nans:
            # Randomly select positions to set to NaN
            selected = np.random.choice(non_nan_positions, n_more_nans, replace=False)
            selected_rows = selected // n_samples
            selected_cols = selected % n_samples
            sparse_data[selected_rows, selected_cols] = np.nan
    
    # If we have too many NaNs, randomly restore some values
    elif actual_nans > n_nans:
        # Find positions that are NaN
        nan_mask = np.isnan(sparse_data)
        nan_positions = np.where(nan_mask.flatten())[0]
        
        # How many NaNs do we need to restore?
        n_restore = actual_nans - n_nans
        
        if len(nan_positions) >= n_restore:
            # Randomly select NaN positions to restore
            selected = np.random.choice(nan_positions, n_restore, replace=False)
            selected_rows = selected // n_samples
            selected_cols = selected % n_samples
            
            # Restore values (use median of non-NaN values for that feature)
            for i, j in zip(selected_rows, selected_cols):
                feature_values = data[i, ~np.isnan(data[i])]
                if len(feature_values) > 0:
                    sparse_data[i, j] = np.median(feature_values)
                else:
                    sparse_data[i, j] = 0.5  # Fallback if all values are NaN
    
    return sparse_data


def simulate_methylation_samples_by_class(model, class_to_generate, X, classes, class_mapping,
                                        n_samples=10, device=None, variability_factor=1.5,
                                        use_individual_samples=True, sparsity_level=0.0,
                                        sparsity_pattern=None, original_data=None):
    """
    Simulate new methylation samples for a specific class with enhanced diversity.
    
    Parameters:
    -----------
    model : ConditionalBetaVAE
        Trained VAE model
    class_to_generate : any
        The class to generate samples for
    X : numpy.ndarray
        Original data (samples as rows)
    classes : array-like
        Original class labels from training
    class_mapping : dict
        Mapping from class labels to numeric indices
    n_samples : int
        Number of samples to generate
    device : torch.device
        Device to run the model on
    variability_factor : float
        Factor to increase variability in latent space
    use_individual_samples : bool
        Whether to use individual encoded samples as starting points
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
    latent_vectors : numpy.ndarray
        Latent vectors used to generate the samples
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
    
    # Get the numeric class index
    if class_to_generate in class_mapping:
        class_idx = class_mapping[class_to_generate]
    else:
        raise ValueError(f"Class {class_to_generate} not found in class mapping")
    
    # Convert to PyTorch tensor
    X_class_tensor = torch.FloatTensor(X_class).to(device)
    class_tensor = torch.LongTensor([class_idx] * X_class.shape[0]).to(device)
    
    # Prepare to store latent vectors
    latent_vectors = np.zeros((n_samples, model.latent_dim))
    
    # Encode samples to get latent space distribution
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(X_class_tensor)
        
        # Two methods for generating diverse samples:
        if use_individual_samples:
            # Method 1: Use individual encoded samples as starting points
            generated_list = []
            
            for i in range(n_samples):
                # Randomly select a real sample encoding as starting point
                sample_idx = np.random.randint(len(class_indices))
                sample_mu = mu[sample_idx]
                sample_logvar = logvar[sample_idx]
                
                # Sample from this distribution with increased variability
                z = model.reparameterize(sample_mu, sample_logvar + torch.log(torch.tensor(variability_factor)))
                
                # Add some random noise to increase diversity
                noise_scale = torch.exp(0.5 * sample_logvar).mean() * 0.4
                z = z + torch.randn_like(z) * noise_scale
                
                # Store the latent vector
                latent_vectors[i] = z.cpu().numpy()
                
                # Decode with class conditioning
                class_idx_tensor = torch.tensor([class_idx], device=device)
                decoded = model.decode(z.unsqueeze(0), class_idx_tensor)
                
                generated_list.append(decoded.squeeze().cpu().numpy())
                
            # Stack all generated samples
            generated = np.stack(generated_list)
            
        else:
            # Method 2: Use class statistics in latent space
            # Calculate mean and std in latent space for this class
            z_mean_avg = mu.mean(dim=0)
            z_std = torch.exp(0.5 * logvar).mean(dim=0) * variability_factor
            
            # Add class-specific variations to increase diversity
            z_variations = []
            for i in range(min(5, len(class_indices))):
                z_variations.append(mu[i] - z_mean_avg)
            
            # Generate samples from class-specific distribution
            class_idx_tensor = torch.LongTensor([class_idx] * n_samples).to(device)
            z_sampled = torch.randn(n_samples, model.latent_dim).to(device)
            z_sampled = z_sampled * z_std + z_mean_avg
            
            # Add sample-specific variations to some samples
            if len(z_variations) > 0:
                for i in range(min(n_samples, len(z_variations))):
                    variation_scale = np.random.uniform(0.3, 0.7)
                    z_sampled[i] = z_sampled[i] + z_variations[i % len(z_variations)] * variation_scale
            
            # Store the latent vectors
            latent_vectors = z_sampled.cpu().numpy()
            
            # Decode to get new samples with class conditioning
            generated = model.decode(z_sampled, class_idx_tensor).cpu().numpy()
    
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
    
    return generated, latent_vectors


def apply_contamination(simulated_samples, contamination_profiles, contamination_proportion,
                        contamination_method='random_weighted', contamination_params=None):
    """
    Apply contamination to simulated methylation samples by mixing in other profiles with enhanced realism.
    
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
        - 'site_specific': Apply different contamination levels to different CpG sites
    contamination_params : dict, optional
        Additional parameters for the contamination method:
        - For 'beta_distribution': 'alpha' and 'beta' parameters
        - For 'random_weighted': 'weights' for different contamination sources
        - For 'site_specific': 'site_bias' to control site-specific contamination bias
        
    Returns:
    --------
    contaminated_samples : numpy.ndarray
        Samples with contamination applied (features as rows)
    contamination_levels : numpy.ndarray
        Actual contamination proportion applied to each sample
    contamination_sources : list
        List of contamination source identifiers used for each sample
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
    
    # Initialize contamination levels and sources
    contamination_levels = np.zeros(n_samples)
    contamination_sources = [[] for _ in range(n_samples)]
    
    # Process contamination profiles
    if isinstance(contamination_profiles, dict):
        # Multiple contamination sources
        contamination_sources_dict = contamination_profiles
        source_keys = list(contamination_sources_dict.keys())
        contamination_sources_list = list(contamination_sources_dict.values())
        n_sources = len(contamination_sources_list)
        
        # Check if all sources have the same number of features
        for source in contamination_sources_list:
            if source.shape[0] != n_features:
                raise ValueError(f"Contamination source has {source.shape[0]} features, "
                                f"but simulated samples have {n_features}")
    else:
        # Single contamination source
        if contamination_profiles.shape[0] != n_features:
            raise ValueError(f"Contamination profile has {contamination_profiles.shape[0]} features, "
                            f"but simulated samples have {n_features}")
        source_keys = ['unknown']
        contamination_sources_list = [contamination_profiles]
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
            source = contamination_sources_list[source_idx]
            source_key = source_keys[source_idx]
            
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
            contamination_sources[i].append(source_key)
            
    elif contamination_method == 'random':
        # Apply random contamination proportion to each sample
        for i in range(n_samples):
            # Randomly select a contamination source if multiple are provided
            source_idx = np.random.randint(n_sources)
            source = contamination_sources_list[source_idx]
            source_key = source_keys[source_idx]
            
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
            contamination_sources[i].append(source_key)
            
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
            sample_sources = []
            for j, source in enumerate(contamination_sources_list):
                # Randomly select a sample from the source if it has multiple samples
                if source.shape[1] > 1:
                    sample_idx = np.random.randint(source.shape[1])
                    contaminant = source[:, sample_idx:sample_idx+1]
                else:
                    contaminant = source
                
                # Add this source's contribution
                source_prop = prop * source_props[j]
                if source_prop > 0.01:  # Only add sources with meaningful contribution
                    mixed_sample += source_prop * contaminant
                    sample_sources.append(source_keys[j])
            
            contaminated_samples[:, i:i+1] = mixed_sample
            contamination_levels[i] = prop
            contamination_sources[i] = sample_sources
            
    elif contamination_method == 'beta_distribution':
        # Sample contamination proportions from a beta distribution
        alpha = contamination_params.get('alpha', 2.0)
        beta = contamination_params.get('beta', 5.0)
        
        for i in range(n_samples):
            # Randomly select a contamination source if multiple are provided
            source_idx = np.random.randint(n_sources)
            source = contamination_sources_list[source_idx]
            source_key = source_keys[source_idx]
            
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
            contamination_sources[i].append(source_key)
    
    elif contamination_method == 'site_specific':
        # Apply different contamination levels to different CpG sites
        # This better models biological reality where some regions are more affected by contamination
        
        # Site bias parameter controls how variable contamination is across sites
        site_bias = contamination_params.get('site_bias', 0.5)
        
        for i in range(n_samples):
            # Overall contamination level for this sample
            max_prop = contamination_proportion[i]
            
            # Select multiple contamination sources
            n_contaminants = min(np.random.randint(1, 4), n_sources)  # 1-3 contaminants per sample
            source_indices = np.random.choice(n_sources, n_contaminants, replace=False)
            
            # Calculate contamination proportions for each source
            source_props = np.random.dirichlet(np.ones(n_contaminants)) * max_prop
            
            # Start with the pure sample
            mixed_sample = np.copy(contaminated_samples[:, i:i+1])
            
            # For each selected source
            sample_sources = []
            for j, source_idx in enumerate(source_indices):
                source = contamination_sources_list[source_idx]
                source_key = source_keys[source_idx]
                
                # Only add sources with meaningful contribution
                if source_props[j] < 0.01:
                    continue
                    
                # Randomly select a sample from the source if it has multiple samples
                if source.shape[1] > 1:
                    sample_idx = np.random.randint(source.shape[1])
                    contaminant = source[:, sample_idx:sample_idx+1]
                else:
                    contaminant = source
                
                # Generate site-specific contamination profiles
                # Some sites are more affected by contamination than others
                site_weights = np.random.beta(2, 5, size=(n_features, 1))
                site_weights = site_weights**site_bias  # Control variability
                
                # Normalize to maintain overall contamination level
                site_weights = site_weights / np.mean(site_weights) * source_props[j]
                
                # Apply site-specific contamination
                mixed_sample = (1 - site_weights) * mixed_sample + site_weights * contaminant
                sample_sources.append(source_key)
            
            contaminated_samples[:, i:i+1] = mixed_sample
            contamination_levels[i] = max_prop
            contamination_sources[i] = sample_sources
    
    # Ensure valid methylation beta values
    contaminated_samples = np.clip(contaminated_samples, 0, 1)
    
    return contaminated_samples, contamination_levels, contamination_sources


def calculate_diversity_metrics(samples):
    """
    Calculate diversity metrics for a set of generated samples.
    
    Parameters:
    -----------
    samples : numpy.ndarray
        Samples to evaluate (features as rows)
        
    Returns:
    --------
    metrics : dict
        Dictionary of diversity metrics
    """
    n_features, n_samples = samples.shape
    metrics = {}
    
    # Average pairwise distance (higher = more diverse)
    if n_samples > 1:
        # Transpose to get samples as rows for pairwise_distances
        samples_t = samples.T
        
        # Calculate pairwise distances
        pairwise_dist = pairwise_distances(samples_t)
        
        # Get upper triangle indices (excluding diagonal)
        upper_tri_idx = np.triu_indices(n_samples, 1)
        
        # Calculate metrics
        metrics['avg_pairwise_distance'] = np.mean(pairwise_dist[upper_tri_idx])
        metrics['min_pairwise_distance'] = np.min(pairwise_dist[upper_tri_idx])
        metrics['max_pairwise_distance'] = np.max(pairwise_dist[upper_tri_idx])
        metrics['std_pairwise_distance'] = np.std(pairwise_dist[upper_tri_idx])
    else:
        # Only one sample, so no pairwise distances
        metrics['avg_pairwise_distance'] = 0.0
        metrics['min_pairwise_distance'] = 0.0
        metrics['max_pairwise_distance'] = 0.0
        metrics['std_pairwise_distance'] = 0.0
    
    # Feature-wise statistics
    metrics['feature_avg_std'] = np.mean(np.std(samples, axis=1))
    metrics['feature_max_std'] = np.max(np.std(samples, axis=1))
    metrics['feature_min_std'] = np.min(np.std(samples, axis=1))
    
    # Distribution of methylation values
    metrics['avg_methylation'] = np.mean(samples)
    metrics['methylation_std'] = np.std(samples)
    
    # Bimodality coefficient (higher = more bimodal, methylation is typically bimodal)
    sample_mean = np.mean(samples)
    sample_var = np.var(samples)
    
    if sample_var > 0:
        # Calculate skewness
        skewness = np.mean(((samples - sample_mean) / np.sqrt(sample_var))**3)
        
        # Calculate kurtosis
        kurtosis = np.mean(((samples - sample_mean) / np.sqrt(sample_var))**4)
        
        # Calculate bimodality coefficient
        bimodality = (skewness**2 + 1) / (kurtosis + 3 * ((n_samples - 1)**2) / ((n_samples - 2) * (n_samples - 3)))
        metrics['bimodality_coefficient'] = bimodality
    else:
        metrics['bimodality_coefficient'] = 0.0
    
    return metrics


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


def visualize_latent_space(model, X, classes, class_mapping, device=None, n_samples=1000, 
                          method='tsne', inverse_class_mapping=None):
    """
    Visualize the latent space of the model using dimensionality reduction.
    
    Parameters:
    -----------
    model : ConditionalBetaVAE
        Trained VAE model
    X : numpy.ndarray
        Data to encode (samples as rows)
    classes : array-like
        Class labels for each sample
    class_mapping : dict
        Mapping from class labels to numeric indices
    device : torch.device, optional
        Device to run the model on
    n_samples : int
        Maximum number of samples to visualize
    method : str
        Dimensionality reduction method ('tsne', 'umap', or 'pca')
    inverse_class_mapping : dict, optional
        Mapping from numeric indices to class labels
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with the visualization
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create inverse class mapping if not provided
    if inverse_class_mapping is None:
        inverse_class_mapping = {v: k for k, v in class_mapping.items()}
    
    # Subsample if needed
    if X.shape[0] > n_samples:
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        X_subset = X[indices]
        classes_subset = [classes[i] for i in indices]
    else:
        X_subset = X
        classes_subset = classes
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_subset).to(device)
    class_indices = np.array([class_mapping[c] for c in classes_subset])
    
    # Encode data to get latent representations
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(X_tensor)
        latent_vectors = mu.cpu().numpy()
    
    # Apply dimensionality reduction
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
        except ImportError:
            print("UMAP not installed. Falling back to PCA.")
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
    else:  # default to PCA
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
    
    # Reduce dimensionality
    embedded = reducer.fit_transform(latent_vectors)
    
    # Create a figure
    plt.figure(figsize=(12, 10))
    
    # Get unique classes
    unique_classes = np.unique(class_indices)
    
    # Create color map
    cmap = plt.cm.get_cmap('tab20', len(unique_classes))
    
    # Plot each class
    for i, class_idx in enumerate(unique_classes):
        mask = class_indices == class_idx
        class_name = inverse_class_mapping[class_idx]
        plt.scatter(embedded[mask, 0], embedded[mask, 1], c=[cmap(i)], 
                   label=f"{class_name}", alpha=0.7, s=50)
    
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Latent Space Visualization ({method.upper()})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"latent_space_{method}.png")
    print(f"Latent space visualization saved to latent_space_{method}.png")
    
    return plt.gcf()


def generate_samples_per_class_with_contamination(df, classes, samples_per_class=10, latent_dim=20, 
                                                hidden_dim=256, epochs=100, sparsity_level=0.1, 
                                                sparsity_pattern='biologically_inspired', 
                                                contamination_labels=None, contamination_proportion=0.0, 
                                                contamination_method='site_specific', 
                                                contamination_params=None, 
                                                variability_factor=1.5,
                                                device=None,
                                                save_model=True,
                                                visualize_results=True,
                                                use_vamp_prior=False):  # Set VampPrior to False by default
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
    variability_factor : float
        Factor to increase variability in latent space sampling
    device : torch.device
        Device to run the model on
    save_model : bool
        Whether to save the trained model
    visualize_results : bool
        Whether to create visualizations of the results
    use_vamp_prior : bool
        Whether to use VampPrior in the loss function
        
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
    inverse_class_mapping = {i: c for i, c in enumerate(unique_classes)}
    
    # Print configuration
    print("=== Configuration ===")
    print(f"Number of features: {df.shape[0]}")
    print(f"Number of samples: {df.shape[1]}")
    print(f"Number of classes: {len(unique_classes)}")
    print(f"Samples per class to generate: {samples_per_class}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Training epochs: {epochs}")
    print(f"Sparsity level: {sparsity_level}")
    print(f"Sparsity pattern: {sparsity_pattern}")
    print(f"Contamination proportion: {contamination_proportion}")
    print(f"Contamination method: {contamination_method}")
    print(f"Device: {device}")
    print("====================")
        
    # Train the advanced VAE
    model, col_means, class_mapping, device = train_advanced_vae(
        df, classes, latent_dim=latent_dim, hidden_dim=hidden_dim, 
        epochs=epochs, device=device, beta_weight_end=0.5,
        use_vamp_prior=use_vamp_prior, consistency_weight=0.2
    )

    # Save the model if requested
    if save_model:
        metadata = {
            'feature_names': df.index.tolist(),
            'col_means': col_means.tolist(),
            'class_mapping': class_mapping,
            'inverse_class_mapping': {str(v): k for k, v in class_mapping.items()},
            'n_classes': len(unique_classes),
            'training_samples': len(classes),
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'description': 'Advanced Conditional VAE model for diverse methylation simulation'
        }
        save_methylation_vae_model(model, "advanced_methylation_vae_model.pth", metadata=metadata)
    
    # Transpose df to get samples as rows
    X = df.T.values
    
    # Visualize latent space if requested
    if visualize_results:
        visualize_latent_space(model, X, classes, class_mapping, device, method='tsne')
        try:
            # Try UMAP if available
            visualize_latent_space(model, X, classes, class_mapping, device, method='umap')
        except:
            pass
    
    # Dictionary to store generated samples for each class
    simulated_data = {}
    latent_vectors = {}
    diversity_metrics = {}
    contamination_data = {}
    
    # Generate pure samples for each class
    print("\n=== Generating samples ===")
    for class_label in unique_classes:
        print(f"Generating {samples_per_class} samples for class '{class_label}'...")
        
        # Generate pure samples with higher diversity
        pure_samples, class_latents = simulate_methylation_samples_by_class(
            model, class_label, X, classes, class_mapping,
            n_samples=samples_per_class, device=device,
            variability_factor=variability_factor,
            use_individual_samples=True,
            sparsity_level=0.0  # No sparsity yet, we'll apply it after contamination
        )
        
        # Store pure samples for potential use as contamination sources
        simulated_data[class_label] = pure_samples
        latent_vectors[class_label] = class_latents
        
        # Calculate diversity metrics
        diversity_metrics[class_label] = calculate_diversity_metrics(pure_samples)
        print(f"  Average pairwise distance: {diversity_metrics[class_label]['avg_pairwise_distance']:.4f}")
        print(f"  Feature average std: {diversity_metrics[class_label]['feature_avg_std']:.4f}")
    
    # Create contamination sources if requested
    if contamination_labels is not None and contamination_proportion > 0:
        contamination_sources = {}
        
        # Check if contamination labels are valid
        for label in contamination_labels:
            if label not in unique_classes:
                raise ValueError(f"Contamination label {label} not found in classes")
            
            # Use the pure simulated samples as contamination sources
            contamination_sources[label] = simulated_data[label]
        
        print("\n=== Applying contamination ===")
        # Apply contamination to each class
        for class_label in unique_classes:
            print(f"Applying contamination to class '{class_label}'...")
            
            # Skip if this class is used only as a contamination source
            if class_label in contamination_labels and class_label not in simulated_data:
                continue
                
            # Apply contamination
            contaminated_samples, contamination_levels, contam_sources = apply_contamination(
                simulated_data[class_label],
                contamination_sources,
                contamination_proportion,
                contamination_method,
                contamination_params
            )
            
            # Store contamination levels
            contamination_data[class_label] = {
                'levels': contamination_levels,
                'sources': contam_sources
            }
            
            # Apply sparsity after contamination
            if sparsity_level > 0:
                print(f"Applying {sparsity_level:.1%} sparsity with '{sparsity_pattern}' pattern...")
                
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
            
            # Calculate diversity metrics after contamination
            #post_diversity = calculate_diversity_metrics(contaminated_samples)
            #print(f"  Post-contamination avg pairwise distance: {post_diversity['avg_pairwise_distance']:.4f}")
            #print(f"  Post-contamination feature avg std: {post_diversity['feature_avg_std']:.4f}")
    else:
        # Apply sparsity to pure samples if no contamination
        if sparsity_level > 0:
            print("\n=== Applying sparsity ===")
            for class_label in unique_classes:
                print(f"Applying {sparsity_level:.1%} sparsity to class '{class_label}' with '{sparsity_pattern}' pattern...")
                
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
            for class_label in unique_classes:
                contamination_data[class_label] = {
                    'levels': np.zeros(simulated_data[class_label].shape[1]),
                    'sources': [[] for _ in range(simulated_data[class_label].shape[1])]
                }
    
    # Prepare data for the combined DataFrame
    feature_names = df.index.tolist()
    
    # Combine all simulated samples into a single matrix
    all_samples_list = []
    sample_metadata = {
        'sample_id': [],
        'class': [],
        'contamination_level': [],
        'contamination_sources': [],
        'original_latent_vector': [],
        'diversity_metrics': []
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
            sample_metadata['contamination_sources'].extend(contamination_data[class_label]['sources'])
        else:
            sample_metadata['contamination_level'].extend([0.0] * n_samples)
            sample_metadata['contamination_sources'].extend([[]] * n_samples)
        
        # Add latent vectors
        for i in range(n_samples):
            if i < len(latent_vectors[class_label]):
                sample_metadata['original_latent_vector'].append(latent_vectors[class_label][i].tolist())
            else:
                sample_metadata['original_latent_vector'].append([])
        
        # Add diversity metrics for each sample (same for all samples in a class)
        class_diversity = calculate_diversity_metrics(samples)
        sample_metadata['diversity_metrics'].extend([class_diversity] * n_samples)
    
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
            'contamination_sources': sample_metadata['contamination_sources'],
            'diversity_metrics': sample_metadata['diversity_metrics']
        }, index=sample_metadata['sample_id'])
        
        # Save latent vectors separately as they can be large
        latent_vectors_df = pd.DataFrame({
            'latent_vector': sample_metadata['original_latent_vector']
        }, index=sample_metadata['sample_id'])
        
        # Save to parquet files
        print("\n=== Saving results ===")
        simulated_df.to_parquet("improved_simulated_methylation_samples.parquet")
        metadata_df.to_parquet("improved_simulated_metadata.parquet")
        latent_vectors_df.to_parquet("simulated_latent_vectors.parquet")
        print("Improved simulated methylation samples and metadata saved.")
        
        # Visualize sample diversity if requested
        if visualize_results:
            try:
                # Create class-colored heatmap of simulated data
                plt.figure(figsize=(15, 10))
                
                # Prepare data for visualization (subset if too large)
                if simulated_df.shape[0] > 1000:
                    viz_samples = simulated_df.iloc[np.random.choice(simulated_df.shape[0], 1000, replace=False)]
                else:
                    viz_samples = simulated_df
                
                # Get class colors
                class_colors = []
                cmap = plt.cm.get_cmap('tab20', len(unique_classes))
                class_color_map = {c: cmap(i) for i, c in enumerate(unique_classes)}
                
                for sample_id in viz_samples.columns:
                    class_label = metadata_df.loc[sample_id, 'class']
                    class_colors.append(class_color_map[class_label])
                
                # Sort classes for better visualization
                class_order = pd.DataFrame({
                    'sample_id': viz_samples.columns,
                    'class': [metadata_df.loc[s, 'class'] for s in viz_samples.columns]
                }).sort_values('class')
                
                # Heatmap
                plt.imshow(
                    viz_samples[class_order['sample_id']].values, 
                    aspect='auto', 
                    cmap='viridis', 
                    interpolation='nearest'
                )
                
                plt.colorbar(label='Methylation β value')
                plt.title('Simulated Methylation Samples')
                plt.xlabel('Samples (colored by class)')
                plt.ylabel('CpG Sites')
                
                # Add class color bar above plot
                ax2 = plt.gca().twiny()
                ax2.set_xlim(plt.gca().get_xlim())
                ax2.set_xticks([])
                
                for i, sample_id in enumerate(class_order['sample_id']):
                    ax2.add_patch(plt.Rectangle(
                        (i - 0.5, -0.5), 
                        1, 
                        0.5, 
                        color=class_color_map[metadata_df.loc[sample_id, 'class']], 
                        clip_on=False, 
                        transform=ax2.transData
                    ))
                
                # Add legend
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color=class_color_map[c], lw=4, label=c)
                    for c in unique_classes
                ]
                plt.legend(
                    handles=legend_elements, 
                    loc='upper center', 
                    bbox_to_anchor=(0.5, 1.15),
                    ncol=min(6, len(unique_classes))
                )
                
                plt.tight_layout()
                plt.savefig('simulated_methylation_heatmap.png')
                print("Heatmap visualization saved to 'simulated_methylation_heatmap.png'")
            except Exception as e:
                print(f"Error creating heatmap visualization: {e}")
        
        return simulated_df, metadata_df, model, diversity_metrics
    else:
        return pd.DataFrame(), pd.DataFrame(), model, {}


# Example usage with improved parameters
if __name__ == "__main__":
    # Load data
    betas = pd.read_parquet("HM450_reference_v3.parquet")
    m = pd.read_parquet("reference_v3.parquet")
    classes = m.loc[:, "entity"].values
    feature_names = betas.index.values
    
    # Generate improved simulated samples
    simulated_df, metadata_df, model, diversity_metrics = generate_samples_per_class_with_contamination(
        betas, classes, 
        samples_per_class=20,  # Increased sample count
        latent_dim=128,         # Increased from 10
        hidden_dim=256,        # Increased from 128
        epochs=100,            # Increased from 50
        sparsity_level=0.0,
        sparsity_pattern='biologically_inspired',
        contamination_labels=None,
        contamination_proportion=0.0,
        contamination_method='site_specific',
        variability_factor=1.5,
        visualize_results=True,
        use_vamp_prior=False   # Disable VampPrior to avoid tensor size errors
    )
    
    print("\n=== Simulation Complete ===")
    print(f"Generated {simulated_df.shape[1]} samples for {len(np.unique(classes))} classes")
    print(f"Average diversity score: {np.mean([m['avg_pairwise_distance'] for m in diversity_metrics.values()]):.4f}")
    
    # Summary of diversity metrics by class
    print("\n=== Diversity Metrics by Class ===")
    for class_label, metrics in diversity_metrics.items():
        print(f"Class '{class_label}':")
        print(f"  Average pairwise distance: {metrics['avg_pairwise_distance']:.4f}")
        print(f"  Feature variation (avg std): {metrics['feature_avg_std']:.4f}")
        print(f"  Bimodality coefficient: {metrics['bimodality_coefficient']:.4f}")