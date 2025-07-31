import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import pairwise_distances


class MethylationCorrectionMLP(nn.Module):
    """
    MLP that monitors and corrects VAE outputs to better match expected class characteristics.
    Works with ConditionalBetaVAE for methylation data simulation.
    """
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.2, num_classes=None, 
                 class_embedding_dim=16, include_latent=True, latent_dim=20):
        super(MethylationCorrectionMLP, self).__init__()
        
        self.include_latent = include_latent
        self.input_dim = input_dim
        
        # Class embedding for conditioning
        if num_classes is not None:
            self.use_class_embedding = True
            self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)
            combined_input_dim = input_dim + class_embedding_dim
            if include_latent:
                combined_input_dim += latent_dim
        else:
            self.use_class_embedding = False
            combined_input_dim = input_dim
            if include_latent:
                combined_input_dim += latent_dim
        
        # Correction network - designed specifically for methylation data
        self.network = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Small adjustment network to fine-tune corrections
        self.adjustment = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid()
        )
        
        # Output normalization to maintain beta value distribution
        self.output_norm = nn.Sigmoid()
    
    def forward(self, x, class_indices=None, latent_vector=None):
        """
        Forward pass through the correction network.
        
        Parameters:
        -----------
        x : torch.Tensor
            VAE output samples to correct (methylation beta values)
        class_indices : torch.Tensor
            Indices of the target classes (optional)
        latent_vector : torch.Tensor
            Latent vectors used to generate the samples (optional)
            
        Returns:
        --------
        corrected_x : torch.Tensor
            Corrected samples
        correction_magnitude : torch.Tensor
            Magnitude of the applied correction
        """
        # Build combined input
        if self.use_class_embedding and class_indices is not None:
            class_embed = self.class_embedding(class_indices)
            if self.include_latent and latent_vector is not None:
                combined = torch.cat([x, class_embed, latent_vector], dim=1)
            else:
                combined = torch.cat([x, class_embed], dim=1)
        else:
            if self.include_latent and latent_vector is not None:
                combined = torch.cat([x, latent_vector], dim=1)
            else:
                combined = x
                
        # Generate correction values
        correction = self.network(combined)
        
        # Create gating mechanism to apply corrections selectively
        # This helps preserve realistic values and prevents over-correction
        gate_input = torch.cat([x, correction], dim=1)
        adjustment_weights = self.adjustment(gate_input)
        
        # Apply weighted correction with skip connection
        # Using residual connection helps maintain original structure
        corrected_x = x + adjustment_weights * correction
        
        # Ensure output stays in valid beta value range with margin
        corrected_x = torch.clamp(corrected_x, 0.001, 0.999)
        
        # Calculate correction magnitude for monitoring
        correction_magnitude = torch.mean(torch.abs(adjustment_weights * correction))
        
        return corrected_x, correction_magnitude


class CorrectedConditionalBetaVAE(nn.Module):
    """
    Wrapper that combines ConditionalBetaVAE with a correction MLP
    for methylation data simulation with improved class adherence.
    """
    def __init__(self, vae_model, correction_mlp):
        super(CorrectedConditionalBetaVAE, self).__init__()
        self.vae = vae_model
        self.correction_mlp = correction_mlp
        
    def forward(self, x, class_indices=None):
        """
        Forward pass through the combined model.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input data
        class_indices : torch.Tensor
            Indices of the target classes
            
        Returns:
        --------
        initial_recon : torch.Tensor
            Initial reconstruction from VAE
        corrected_recon : torch.Tensor
            Corrected reconstruction
        mu : torch.Tensor
            Mean of latent distribution
        logvar : torch.Tensor
            Log variance of latent distribution
        correction_magnitude : torch.Tensor
            Magnitude of the applied correction
        """
        # VAE forward pass
        initial_recon, mu, logvar = self.vae(x, class_indices)
        
        # Sample latent vector for correction
        z = self.vae.reparameterize(mu, logvar)
        
        # Apply correction
        corrected_recon, correction_magnitude = self.correction_mlp(
            initial_recon, class_indices, z
        )
        
        return initial_recon, corrected_recon, mu, logvar, correction_magnitude
    
    def generate(self, n_samples, class_indices=None, z=None, return_latent=False):
        """
        Generate samples with correction.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        class_indices : torch.Tensor
            Indices of the target classes
        z : torch.Tensor
            Optional latent vectors to use
        return_latent : bool
            Whether to return the latent vectors
            
        Returns:
        --------
        corrected_samples : torch.Tensor
            Corrected generated samples
        initial_samples : torch.Tensor
            Initial generated samples (before correction)
        z : torch.Tensor
            Latent vectors (if return_latent=True)
        """
        # Generate latent vectors if not provided
        if z is None:
            z = torch.randn(n_samples, self.vae.latent_dim, device=next(self.parameters()).device)
        
        # Generate samples from latent vectors
        initial_samples = self.vae.decode(z, class_indices)
        
        # Apply correction
        corrected_samples, _ = self.correction_mlp(
            initial_samples, class_indices, z
        )
        
        if return_latent:
            return corrected_samples, initial_samples, z
        else:
            return corrected_samples, initial_samples


def class_fidelity_loss(samples, class_indices, reference_stats, weight=1.0):
    """
    Loss function to measure how well samples match class-specific statistics.
    
    Parameters:
    -----------
    samples : torch.Tensor
        Batch of samples to evaluate
    class_indices : torch.Tensor
        Class indices for each sample
    reference_stats : dict
        Dictionary with class-specific statistics
        (means, variances, correlations)
    weight : float
        Weight for the loss term
        
    Returns:
    --------
    loss : torch.Tensor
        Weighted class fidelity loss
    """
    batch_size = samples.size(0)
    device = samples.device
    total_loss = torch.tensor(0.0, device=device)
    
    # Process each class in the batch separately
    unique_classes = torch.unique(class_indices)
    for cls_idx in unique_classes:
        # Get samples from this class
        cls_mask = (class_indices == cls_idx)
        cls_samples = samples[cls_mask]
        n_cls_samples = cls_samples.size(0)
        
        if n_cls_samples == 0:
            continue
            
        cls_idx_int = cls_idx.item()
        if str(cls_idx_int) not in reference_stats:
            continue
        
        # Get reference statistics for this class
        ref_mean = torch.tensor(reference_stats[str(cls_idx_int)]['mean'], device=device)
        ref_std = torch.tensor(reference_stats[str(cls_idx_int)]['std'], device=device)
        
        # Calculate current statistics
        cls_mean = torch.mean(cls_samples, dim=0)
        cls_std = torch.std(cls_samples, dim=0) + 1e-6  # Avoid division by zero
        
        # Mean adherence loss (using normalized difference)
        mean_loss = F.mse_loss(cls_mean, ref_mean)
        
        # Standard deviation adherence loss (using relative difference)
        std_loss = F.mse_loss(cls_std / ref_std, torch.ones_like(cls_std))
        
        # Optional: Add correlation structure adherence if available
        # This is more complex but would ensure correlation patterns are preserved
        
        # Weight the loss by the number of samples in this class
        weighted_class_loss = (mean_loss + std_loss) * (n_cls_samples / batch_size)
        total_loss += weighted_class_loss
    
    return total_loss * weight


class MetadataGenerator:
    """
    Helper class to compute and store class-specific statistics
    for the correction network.
    """
    def __init__(self, dataloader, model, num_classes):
        self.reference_stats = {}
        self.model = model
        self.num_classes = num_classes
        self.compute_class_statistics(dataloader)
    
    def compute_class_statistics(self, dataloader):
        """Compute statistics for each class"""
        # Initialize storage
        class_samples = {str(i): [] for i in range(self.num_classes)}
        
        # Collect samples by class
        device = next(self.model.parameters()).device
        self.model.eval()
        
        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(device)
                labels = labels.to(device)
                
                # Forward pass
                recon, _, _ = self.model.vae(data, labels)
                
                # Store reconstructions by class
                for cls_idx in range(self.num_classes):
                    cls_mask = (labels == cls_idx)
                    if torch.any(cls_mask):
                        cls_recon = recon[cls_mask].cpu().numpy()
                        class_samples[str(cls_idx)].append(cls_recon)
        
        # Compute statistics for each class
        for cls_idx in range(self.num_classes):
            cls_str = str(cls_idx)
            if not class_samples[cls_str]:
                continue
                
            # Combine samples for this class
            cls_all_samples = np.vstack(class_samples[cls_str])
            
            # Compute basic statistics
            mean = np.mean(cls_all_samples, axis=0)
            std = np.std(cls_all_samples, axis=0) + 1e-6  # Avoid division by zero
            
            # Compute correlation structure (for selected features)
            # Using a subset for efficiency
            n_features = len(mean)
            if n_features > 1000:
                # Use random subset of features for correlation
                feature_subset = np.random.choice(n_features, 1000, replace=False)
                subset_data = cls_all_samples[:, feature_subset]
                try:
                    # Calculate correlation matrix for subset
                    corr_matrix = np.corrcoef(subset_data.T)
                    # Save important structure using eigendecomposition
                    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
                    # Keep top components
                    top_k = 50
                    top_eigenvalues = eigenvalues[-top_k:]
                    top_eigenvectors = eigenvectors[:, -top_k:]
                    
                    self.reference_stats[cls_str] = {
                        'mean': mean,
                        'std': std,
                        'feature_subset': feature_subset,
                        'top_eigenvalues': top_eigenvalues,
                        'top_eigenvectors': top_eigenvectors
                    }
                except np.linalg.LinAlgError:
                    # Fallback if correlation calculation fails
                    self.reference_stats[cls_str] = {
                        'mean': mean,
                        'std': std
                    }
            else:
                # For smaller feature sets, just store mean and std
                self.reference_stats[cls_str] = {
                    'mean': mean,
                    'std': std
                }
                
    def get_reference_stats(self):
        """Return the computed reference statistics"""
        return self.reference_stats


def train_correction_mlp(vae_model, correction_mlp, train_loader, val_loader=None,
                        epochs=50, lr=0.0001, device=None, class_fidelity_weight=1.0,
                        reconstruction_weight=0.5):
    """
    Train the correction MLP to improve class fidelity of VAE outputs.
    
    Parameters:
    -----------
    vae_model : ConditionalBetaVAE
        Pretrained VAE model
    correction_mlp : MethylationCorrectionMLP
        Correction network to train
    train_loader : DataLoader
        DataLoader for training data
    val_loader : DataLoader
        DataLoader for validation data (optional)
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    device : torch.device
        Device to use for training
    class_fidelity_weight : float
        Weight for class fidelity loss
    reconstruction_weight : float
        Weight for reconstruction loss
        
    Returns:
    --------
    combined_model : CorrectedConditionalBetaVAE
        Combined model with trained correction network
    training_history : dict
        Dictionary with training metrics
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create combined model
    combined_model = CorrectedConditionalBetaVAE(vae_model, correction_mlp).to(device)
    
    # Freeze VAE parameters
    for param in combined_model.vae.parameters():
        param.requires_grad = False
    
    # Create optimizer for correction MLP only
    optimizer = optim.Adam(combined_model.correction_mlp.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Compute class statistics for reference
    metadata_gen = MetadataGenerator(train_loader, vae_model, vae_model.num_classes)
    reference_stats = metadata_gen.get_reference_stats()
    
    # Initialize training history
    history = {
        'total_loss': [],
        'recon_loss': [],
        'fidelity_loss': [],
        'correction_magnitude': []
    }
    
    # Training loop
    for epoch in range(epochs):
        combined_model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_fidelity_loss = 0.0
        total_correction_magnitude = 0.0
        batch_count = 0
        
        for batch_idx, (data, class_indices) in enumerate(train_loader):
            # Move data to device
            data = data.to(device)
            class_indices = class_indices.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # VAE forward pass (without gradients for efficiency)
            with torch.no_grad():
                initial_recon, mu, logvar = vae_model(data, class_indices)
                z = vae_model.reparameterize(mu, logvar)
            
            # Correction MLP forward pass
            corrected_recon, correction_magnitude = correction_mlp(
                initial_recon.detach(), class_indices, z.detach()
            )
            
            # Reconstruction loss (corrected samples should still match original data)
            recon_loss = F.mse_loss(corrected_recon, data)
            
            # Class fidelity loss (corrected samples should match class statistics)
            fidelity_loss = class_fidelity_loss(
                corrected_recon, class_indices, reference_stats, 
                weight=class_fidelity_weight
            )
            
            # Combined loss
            loss = reconstruction_weight * recon_loss + class_fidelity_weight * fidelity_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_fidelity_loss += fidelity_loss.item()
            total_correction_magnitude += correction_magnitude.item()
            batch_count += 1
        
        # Compute averages
        avg_loss = total_loss / batch_count
        avg_recon_loss = total_recon_loss / batch_count
        avg_fidelity_loss = total_fidelity_loss / batch_count
        avg_correction_magnitude = total_correction_magnitude / batch_count
        
        # Update history
        history['total_loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon_loss)
        history['fidelity_loss'].append(avg_fidelity_loss)
        history['correction_magnitude'].append(avg_correction_magnitude)
        
        # Validation if provided
        if val_loader is not None:
            val_loss = validate_correction_mlp(
                combined_model, val_loader, reference_stats,
                class_fidelity_weight, reconstruction_weight, device
            )
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                  f"Recon: {avg_recon_loss:.4f}, Fidelity: {avg_fidelity_loss:.4f}, "
                  f"Correction: {avg_correction_magnitude:.4f}, Val Loss: {val_loss:.4f}")
            
            # Update learning rate based on validation loss
            scheduler.step(val_loss)
        else:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                  f"Recon: {avg_recon_loss:.4f}, Fidelity: {avg_fidelity_loss:.4f}, "
                  f"Correction: {avg_correction_magnitude:.4f}")
            
            # Update learning rate based on training loss
            scheduler.step(avg_loss)
    
    return combined_model, history


def validate_correction_mlp(combined_model, val_loader, reference_stats,
                           class_fidelity_weight, reconstruction_weight, device):
    """
    Validate the correction MLP.
    
    Parameters:
    -----------
    combined_model : CorrectedConditionalBetaVAE
        Combined model with correction network
    val_loader : DataLoader
        DataLoader for validation data
    reference_stats : dict
        Dictionary with class-specific statistics
    class_fidelity_weight : float
        Weight for class fidelity loss
    reconstruction_weight : float
        Weight for reconstruction loss
    device : torch.device
        Device to use for validation
        
    Returns:
    --------
    avg_loss : float
        Average validation loss
    """
    combined_model.eval()
    total_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, (data, class_indices) in enumerate(val_loader):
            # Move data to device
            data = data.to(device)
            class_indices = class_indices.to(device)
            
            # Get initial reconstruction and latent vectors
            initial_recon, mu, logvar = combined_model.vae(data, class_indices)
            z = combined_model.vae.reparameterize(mu, logvar)
            
            # Apply correction
            corrected_recon, _ = combined_model.correction_mlp(
                initial_recon, class_indices, z
            )
            
            # Reconstruction loss
            recon_loss = F.mse_loss(corrected_recon, data)
            
            # Class fidelity loss
            fidelity_loss = class_fidelity_loss(
                corrected_recon, class_indices, reference_stats, 
                weight=class_fidelity_weight
            )
            
            # Combined loss
            loss = reconstruction_weight * recon_loss + class_fidelity_weight * fidelity_loss
            
            total_loss += loss.item()
            batch_count += 1
    
    return total_loss / batch_count


def evaluate_class_fidelity(model, test_loader, device=None):
    """
    Evaluate how well the model's outputs match expected class characteristics.
    
    Parameters:
    -----------
    model : CorrectedConditionalBetaVAE
        Combined model with correction network
    test_loader : DataLoader
        DataLoader for test data
    device : torch.device
        Device to use for evaluation
        
    Returns:
    --------
    metrics : dict
        Dictionary with evaluation metrics
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    # Initialize class-specific collections
    class_data = {}
    class_initial_recon = {}
    class_corrected_recon = {}
    
    with torch.no_grad():
        for batch_idx, (data, class_indices) in enumerate(test_loader):
            # Move data to device
            data = data.to(device)
            class_indices = class_indices.to(device)
            
            # Forward pass
            initial_recon, corrected_recon, mu, logvar, _ = model(data, class_indices)
            
            # Store results by class
            for cls_idx in torch.unique(class_indices):
                cls_mask = (class_indices == cls_idx)
                cls_str = str(cls_idx.item())
                
                # Get data for this class
                cls_data = data[cls_mask].cpu().numpy()
                cls_initial = initial_recon[cls_mask].cpu().numpy()
                cls_corrected = corrected_recon[cls_mask].cpu().numpy()
                
                # Initialize if first batch with this class
                if cls_str not in class_data:
                    class_data[cls_str] = []
                    class_initial_recon[cls_str] = []
                    class_corrected_recon[cls_str] = []
                
                # Append batch data
                class_data[cls_str].append(cls_data)
                class_initial_recon[cls_str].append(cls_initial)
                class_corrected_recon[cls_str].append(cls_corrected)
    
    # Compute metrics
    metrics = {}
    
    for cls_str in class_data:
        # Combine batches
        orig_data = np.vstack(class_data[cls_str])
        initial_recon = np.vstack(class_initial_recon[cls_str])
        corrected_recon = np.vstack(class_corrected_recon[cls_str])
        
        # Compute mean squared error
        initial_mse = np.mean((orig_data - initial_recon) ** 2)
        corrected_mse = np.mean((orig_data - corrected_recon) ** 2)
        
        # Compute correlation between features
        feature_corr_orig = np.corrcoef(orig_data.T)
        feature_corr_initial = np.corrcoef(initial_recon.T)
        feature_corr_corrected = np.corrcoef(corrected_recon.T)
        
        # Handle NaNs in correlation matrices
        feature_corr_orig = np.nan_to_num(feature_corr_orig)
        feature_corr_initial = np.nan_to_num(feature_corr_initial)
        feature_corr_corrected = np.nan_to_num(feature_corr_corrected)
        
        # Compute correlation structure preservation
        corr_diff_initial = np.mean((feature_corr_orig - feature_corr_initial) ** 2)
        corr_diff_corrected = np.mean((feature_corr_orig - feature_corr_corrected) ** 2)
        
        # Store metrics
        metrics[cls_str] = {
            'initial_mse': initial_mse,
            'corrected_mse': corrected_mse,
            'mse_improvement': initial_mse - corrected_mse,
            'corr_diff_initial': corr_diff_initial,
            'corr_diff_corrected': corr_diff_corrected,
            'corr_improvement': corr_diff_initial - corr_diff_corrected
        }
    
    return metrics


def create_and_train_correction_mlp(vae_model, df, classes, 
                                   hidden_dim=128, batch_size=32, epochs=50, lr=0.0001,
                                   device=None, class_fidelity_weight=1.0,
                                   include_latent=True):
    """
    Create and train a correction MLP for a trained VAE.
    
    Parameters:
    -----------
    vae_model : ConditionalBetaVAE
        Pretrained VAE model
    df : pandas.DataFrame
        DataFrame with features as rows and samples as columns
    classes : array-like
        Array of class labels for each sample
    hidden_dim : int
        Hidden dimension of the correction MLP
    batch_size : int
        Batch size for training
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    device : torch.device
        Device to use for training
    class_fidelity_weight : float
        Weight for class fidelity loss
    include_latent : bool
        Whether to include latent vectors in correction
        
    Returns:
    --------
    combined_model : CorrectedConditionalBetaVAE
        Combined model with trained correction network
    """
    
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Transpose dataframe to have samples as rows and features as columns
    X = df.T.values
    
    # Create class mapping from original classes to numeric indices
    unique_classes = np.unique(classes)
    class_mapping = {c: i for i, c in enumerate(unique_classes)}
    class_indices = np.array([class_mapping[c] for c in classes])
    
    # Handle NaN values in the data
    print(f"Data contains {np.isnan(X).sum()} NaN values out of {X.size}")
    
    # Replace NaNs with column means (feature-wise imputation)
    col_means = np.nanmean(X, axis=0)
    
    # Find indices of NaN values
    nan_indices = np.isnan(X)
    
    # Replace NaNs with column means
    X[nan_indices] = np.take(col_means, np.where(nan_indices)[1])
    
    # Ensure values are in valid range for numerical stability
    X = np.clip(X, 1e-6, 1 - 1e-6)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X).to(device)
    class_tensor = torch.LongTensor(class_indices).to(device)
    
    # Create dataset and data loader
    dataset = TensorDataset(X_tensor, class_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create correction MLP
    correction_mlp = MethylationCorrectionMLP(
        input_dim=vae_model.input_dim,
        hidden_dim=hidden_dim,
        num_classes=vae_model.num_classes,
        include_latent=include_latent,
        latent_dim=vae_model.latent_dim
    ).to(device)
    
    # Train correction MLP
    combined_model, history = train_correction_mlp(
        vae_model, correction_mlp, train_loader, val_loader=None,
        epochs=epochs, lr=lr, device=device,
        class_fidelity_weight=class_fidelity_weight
    )
    
    return combined_model, history, class_mapping


# Demo: Using the correction MLP with the VAE
def generate_corrected_samples(combined_model, n_samples=10, class_indices=None, device=None):
    """
    Generate samples with the combined model.
    
    Parameters:
    -----------
    combined_model : CorrectedConditionalBetaVAE
        Combined model with correction network
    n_samples : int
        Number of samples to generate
    class_indices : torch.Tensor
        Indices of the target classes
    device : torch.device
        Device to use for generation
        
    Returns:
    --------
    corrected_samples : numpy.ndarray
        Corrected generated samples
    initial_samples : numpy.ndarray
        Initial generated samples (before correction)
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    combined_model.eval()
    
    with torch.no_grad():
        # Generate random latent vectors
        z = torch.randn(n_samples, combined_model.vae.latent_dim, device=device)
        
        # Generate samples
        corrected_samples, initial_samples = combined_model.generate(
            n_samples, class_indices, z
        )
        
        # Convert to numpy
        corrected_samples = corrected_samples.cpu().numpy()
        initial_samples = initial_samples.cpu().numpy()
    
    return corrected_samples, initial_samples