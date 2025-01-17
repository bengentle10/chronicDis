import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

class DiseaseDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DiseasePredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.scaler = StandardScaler()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def save_model(self, path):
        """Save the model and scaler."""
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state and scaler
        checkpoint = {
            'model_state': self.state_dict(),
            'scaler_state': self.scaler,
            'history': self.history,
            'input_size': self.network[0].in_features
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path, device='cpu'):
        """Load a saved model."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(input_size=checkpoint['input_size'])
        model.load_state_dict(checkpoint['model_state'])
        model.scaler = checkpoint['scaler_state']
        model.history = checkpoint['history']
        model.to(device)
        return model

    def prepare_features(self, df):
        """Extract features from disease data."""
        features = pd.DataFrame()
        
        # Basic features
        features['parent_count'] = df['parents'].str.len()
        features['child_count'] = df['children'].str.len()
        features['has_description'] = df['description'].notna().astype(int)
        features['synonym_count'] = df['synonyms'].str.len().fillna(0)
        features['has_dbXRefs'] = df['dbXRefs'].str.len() > 0
        
        # Ontology features
        features['is_leaf'] = df['ontology'].apply(lambda x: int(x.get('leaf', False)))
        features['is_therapeutic'] = df['ontology'].apply(
            lambda x: int(x.get('isTherapeuticArea', False))
        )
        
        return features

    def prepare_data(self, features, labels, batch_size=32):
        """Prepare data loaders for training."""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create datasets
        train_dataset = DiseaseDataset(X_train_scaled, y_train)
        val_dataset = DiseaseDataset(X_val_scaled, y_val)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        return train_loader, val_loader
        
    def forward(self, x):
        return self.network(x)
        
    def train_model(self, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
        """Train the model using the provided data loaders."""
        self.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        for epoch in tqdm(range(epochs), desc="Training"):
            # Training phase
            self.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = self(features).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
            
            # Validation phase
            self.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = self(features).squeeze()
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
            
            # Record metrics
            self.history['train_loss'].append(train_loss / len(train_loader))
            self.history['val_loss'].append(val_loss / len(val_loader))
            self.history['train_acc'].append(train_correct / train_total)
            self.history['val_acc'].append(val_correct / val_total)
            
        return self.history
