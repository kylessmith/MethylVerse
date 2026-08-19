import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        input_size = 100000

        # Define the MLP architecture
        layers = [
            nn.Sequential(nn.Linear(input_size, 512), nn.Sigmoid(), nn.Dropout(0.5)),
            nn.Sequential(nn.Linear(512, 256), nn.Sigmoid(), nn.Dropout(0.3)),
            nn.Sequential(nn.Linear(256, 128), nn.Sigmoid(), nn.Dropout(0.3))
        ]

        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Linear(128, 97)

        self.apply(init_weights)  # Apply weight initialization

    def forward(self, x, extract_features=False):
        emb = self.encoder(x)
        p = self.decoder(emb)
        return emb if extract_features else p


class EnsembleModel(nn.Module):
    def __init__(self, mlp_model1, mlp_model2, mlp_model3, num_classes=97, device=None):
        super(EnsembleModel, self).__init__()
        
        self.mlp1 = mlp_model1
        self.mlp2 = mlp_model2
        self.mlp3 = mlp_model3
        
        combined_feature_size = 128 * 3  # Concatenating three 128-dim feature vectors
        self.bn = nn.BatchNorm1d(combined_feature_size)
        self.fc_final = nn.Linear(combined_feature_size, num_classes)

        # Assign the device
        self.device = device or next(self.mlp1.parameters()).device

    def forward(self, mlp_input1, mlp_input2, mlp_input3):
        mlp_features1 = self.mlp1(mlp_input1, extract_features=True)
        mlp_features2 = self.mlp2(mlp_input2, extract_features=True)
        mlp_features3 = self.mlp3(mlp_input3, extract_features=True)

        combined_features = torch.cat((mlp_features1, mlp_features2, mlp_features3), dim=1)
        
        if combined_features.shape[0] > 1:  # Avoid batch norm issues with batch size 1
            combined_features = self.bn(combined_features)

        output = self.fc_final(combined_features)
        return output  # Returning raw logits for CrossEntropyLoss

    def train_step(self, batch1, batch2, batch3, criterion, optimizer):
        """Trains the ensemble model on a batch of data."""
        self.train()
        
        x1, y = batch1['x'].to(self.device), batch1['y'].to(self.device)
        x2 = batch2['x'].to(self.device)
        x3 = batch3['x'].to(self.device)

        optimizer.zero_grad()
        predictions = self.forward(x1, x2, x3)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

        losses = {'loss.global': loss.item(), 'loss.ce': loss.item()}
        return losses, predictions.detach()

    def evaluate(self, batch1, batch2, batch3, metric):
        """Evaluates the ensemble model using balanced accuracy."""
        self.eval()
        with torch.no_grad():
            x1, y = batch1['x'].to(self.device), batch1['y'].to(self.device)
            x2 = batch2['x'].to(self.device)
            x3 = batch3['x'].to(self.device)

            predictions = self.forward(x1, x2, x3)
            acc = metric(y.cpu().numpy(), predictions.argmax(dim=-1).cpu().numpy())

            return {'balanced_accuracy': acc}

    def predict_step(self, batch1, batch2, batch3):
        """Performs inference on a batch of data."""
        self.eval()
        with torch.no_grad():
            x1 = batch1['x'].to(self.device)
            x2 = batch2['x'].to(self.device)
            x3 = batch3['x'].to(self.device)

            logits = self.forward(x1, x2, x3)
            probabilities = F.softmax(logits, dim=-1)  # Use softmax for probabilities
            predictions = probabilities.argmax(dim=-1)

        return probabilities, predictions

    def save(self, path):
        """Saves the model state and submodels."""
        checkpoint = {
            'ensemble_state_dict': self.state_dict(),
            'mlp1_state_dict': self.mlp1.state_dict(),
            'mlp2_state_dict': self.mlp2.state_dict(),
            'mlp3_state_dict': self.mlp3.state_dict(),
            'num_classes': self.fc_final.out_features,
            'combined_feature_size': self.fc_final.in_features
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path, device=None):
        """Loads the model state from a file."""
        checkpoint = torch.load(path, map_location=device)

        mlp1 = MLP()
        mlp2 = MLP()
        mlp3 = MLP()

        mlp1.load_state_dict(checkpoint['mlp1_state_dict'])
        mlp2.load_state_dict(checkpoint['mlp2_state_dict'])
        mlp3.load_state_dict(checkpoint['mlp3_state_dict'])

        model = cls(mlp1, mlp2, mlp3, num_classes=checkpoint['num_classes'], device=device)
        model.load_state_dict(checkpoint['ensemble_state_dict'])

        print(f"Model loaded from {path}")
        return model