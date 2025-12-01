"""
AstroGuard ML Model Training
=============================
Train LSTM model for satellite collision prediction.

Model Architecture:
- Input: Sequences of orbital features (sequence_length, n_features)
- LSTM Layer 1: 64 units with dropout (0.2)
- LSTM Layer 2: 32 units with dropout (0.2)
- Dense Output: 1 unit (minimum distance prediction)
- Loss: MSE
- Optimizer: Adam
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import os
from datetime import datetime

try:
    from ml.preprocessing import OrbitDataPreprocessor, create_satellite_pairs
except ImportError:
    from preprocessing import OrbitDataPreprocessor, create_satellite_pairs


class CollisionPredictor:
    """LSTM-based model for predicting satellite collision risks."""
    
    def __init__(self, sequence_length: int, n_features: int):
        """
        Initialize collision predictor.
        
        Args:
            sequence_length: Number of timesteps in input sequence
            n_features: Number of features per timestep
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.model_path = Path('models/collision_predictor.h5')
        self.config_path = Path('models/model_config.json')
    
    def build_model(self):
        """Build LSTM neural network with increased capacity."""
        print("\nBuilding enhanced LSTM model architecture...")
        
        model = models.Sequential([
            # First LSTM layer - increased units
            layers.LSTM(128, return_sequences=True, 
                       input_shape=(self.sequence_length, self.n_features),
                       name='lstm_1'),
            layers.Dropout(0.3, name='dropout_1'),
            
            # Second LSTM layer - increased units
            layers.LSTM(64, return_sequences=True, name='lstm_2'),
            layers.Dropout(0.3, name='dropout_2'),
            
            # Third LSTM layer for more depth
            layers.LSTM(32, return_sequences=False, name='lstm_3'),
            layers.Dropout(0.2, name='dropout_3'),
            
            # Dense layers with batch normalization
            layers.Dense(64, activation='relu', name='dense_1'),
            layers.BatchNormalization(name='bn_1'),
            layers.Dropout(0.2, name='dropout_4'),
            
            layers.Dense(32, activation='relu', name='dense_2'),
            layers.Dense(1, activation='linear', name='output')
        ])
        
        # Compile model with learning rate schedule
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
        print("\nModel Summary:")
        print("="*60)
        self.model.summary()
        print("="*60)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 32):
        """
        Train the model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        print(f"\nTraining model for {epochs} epochs...")
        print(f"  Train samples: {len(X_train)}")
        print(f"  Val samples: {len(X_val)}")
        
        # Callbacks with improved patience
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        return history
    
    def save_model(self):
        """Save model and configuration."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model in Keras format (not H5)
        keras_path = self.model_path.with_suffix('.keras')
        self.model.save(str(keras_path))
        
        # Also save in H5 for compatibility
        try:
            self.model.save(str(self.model_path))
        except:
            pass
        
        # Save configuration
        config = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'model_path': str(keras_path),
            'created_at': datetime.now().isoformat()
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[OK] Model saved to {keras_path}")
        print(f"[OK] Config saved to {self.config_path}")
    
    def load_model(self):
        """Load model from file."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = keras.models.load_model(str(self.model_path))
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        self.sequence_length = config['sequence_length']
        self.n_features = config['n_features']
        
        print(f"[OK] Model loaded from {self.model_path}")


def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print(" " * 20 + "AstroGuard ML Training Pipeline")
    print("="*70)
    
    # Step 1: Generate synthetic data
    print("\n[Step 1] Generating synthetic satellite data...")
    preprocessor = OrbitDataPreprocessor()
    satellites = preprocessor.generate_synthetic_dataset(num_satellites=40)
    
    # Step 2: Create satellite pairs
    print("\n[Step 2] Creating satellite pairs for analysis...")
    satellite_pairs = create_satellite_pairs(satellites, max_pairs=50)
    
    # Step 3: Prepare training data
    print("\n[Step 3] Preparing training data (this may take a few minutes)...")
    X, y = preprocessor.prepare_training_data(satellite_pairs)
    
    if len(X) == 0:
        print("\n[ERROR] No training samples generated!")
        return
    
    # Step 4: Split data
    print("\n[Step 4] Splitting data into train/val/test sets...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.18, random_state=42)
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # Step 5: Build model
    print("\n[Step 5] Building LSTM model...")
    predictor = CollisionPredictor(
        sequence_length=X.shape[1],
        n_features=X.shape[2]
    )
    predictor.build_model()
    
    # Step 6: Train model
    print("\n[Step 6] Training model (this will take several minutes)...")
    history = predictor.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32
    )
    
    # Step 7: Evaluate
    print("\n[Step 7] Evaluating model on test set...")
    test_loss, test_mae = predictor.model.evaluate(X_test, y_test, verbose=0)
    y_pred = predictor.model.predict(X_test, verbose=0).flatten()
    
    print(f"\n[OK] Test MAE: {test_mae:.4f} km")
    print(f"[OK] Test RMSE: {np.sqrt(np.mean((y_test - y_pred)**2)):.4f} km")
    print(f"[OK] Accuracy within 5 km: {np.mean(np.abs(y_test - y_pred) < 5) * 100:.1f}%")
    print(f"[OK] Accuracy within 10 km: {np.mean(np.abs(y_test - y_pred) < 10) * 100:.1f}%")
    
    # Step 8: Save model and artifacts
    print("\n[Step 8] Saving model and artifacts...")
    predictor.save_model()
    preprocessor.save_scaler()
    
    # Save training metrics
    metrics = {
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'test_rmse': float(np.sqrt(np.mean((y_test - y_pred)**2))),
        'acc_5km': float(np.mean(np.abs(y_test - y_pred) < 5)),
        'acc_10km': float(np.mean(np.abs(y_test - y_pred) < 10)),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test)
    }
    
    metrics_path = 'models/training_metrics.json'
    os.makedirs('models', exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Saved metrics to {metrics_path}")
    
    print("\n" + "="*70)
    print(" " * 25 + "TRAINING COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
