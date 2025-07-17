"""
Deep Learning Models for Stock Market Analysis
Implements LSTM, Transformer, and ensemble models for pattern recognition and prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Attention,
    MultiHeadAttention, LayerNormalization, Input, Concatenate,
    Conv1D, MaxPooling1D, GlobalMaxPooling1D, Embedding
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any
import joblib
from loguru import logger

from ..config.config import Config


class StockMarketLSTM:
    """LSTM model for stock market prediction"""
    
    def __init__(self, lookback_window: int = 60, prediction_horizon: int = 1):
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        
    def build_model(self, n_features: int, n_outputs: int = 1) -> Model:
        """Build LSTM architecture"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.lookback_window, n_features)),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            
            Dense(16, activation='relu'),
            Dense(n_outputs, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def prepare_sequences(self, data: pd.DataFrame, target_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for LSTM"""
        # Scale features
        feature_cols = [col for col in data.columns if col not in target_cols]
        features_scaled = self.feature_scaler.fit_transform(data[feature_cols])
        
        # Scale targets
        targets_scaled = self.scaler.fit_transform(data[target_cols])
        
        X, y = [], []
        
        for i in range(self.lookback_window, len(data) - self.prediction_horizon + 1):
            # Features sequence
            X.append(features_scaled[i-self.lookback_window:i])
            
            # Target (future values)
            if self.prediction_horizon == 1:
                y.append(targets_scaled[i])
            else:
                y.append(targets_scaled[i:i+self.prediction_horizon])
        
        return np.array(X), np.array(y)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray, epochs: int = 100) -> Dict:
        """Train the LSTM model"""
        
        if self.model is None:
            self.build_model(X_train.shape[2], y_train.shape[1] if len(y_train.shape) > 1 else 1)
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(patience=10, factor=0.5),
            ModelCheckpoint('best_lstm_model.h5', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X)
        
        # Inverse transform predictions
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        
        return self.scaler.inverse_transform(predictions)


class StockMarketTransformer:
    """Transformer model for stock market analysis"""
    
    def __init__(self, seq_length: int = 60, d_model: int = 128, num_heads: int = 8, num_layers: int = 4):
        self.seq_length = seq_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.model = None
        self.scaler = MinMaxScaler()
        
    def positional_encoding(self, length: int, depth: int) -> tf.Tensor:
        """Create positional encoding"""
        depth = depth / 2
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :] / depth
        
        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        
        pos_encoding = np.concatenate([
            np.sin(angle_rads), np.cos(angle_rads)
        ], axis=-1)
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def build_model(self, n_features: int, n_outputs: int = 1) -> Model:
        """Build Transformer architecture"""
        inputs = Input(shape=(self.seq_length, n_features))
        
        # Linear projection to d_model
        x = Dense(self.d_model)(inputs)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding(self.seq_length, self.d_model)
        x = x + pos_encoding
        
        # Transformer blocks
        for _ in range(self.num_layers):
            # Multi-head attention
            attn_output = MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads,
                dropout=0.1
            )(x, x)
            
            # Add & Norm
            x = LayerNormalization()(x + attn_output)
            
            # Feed forward
            ff_output = Dense(self.d_model * 4, activation='relu')(x)
            ff_output = Dropout(0.1)(ff_output)
            ff_output = Dense(self.d_model)(ff_output)
            
            # Add & Norm
            x = LayerNormalization()(x + ff_output)
        
        # Global average pooling
        x = tf.reduce_mean(x, axis=1)
        
        # Final dense layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(n_outputs, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, data: pd.DataFrame, target_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for Transformer"""
        feature_cols = [col for col in data.columns if col not in target_cols]
        
        # Scale data
        features_scaled = self.scaler.fit_transform(data[feature_cols])
        targets = data[target_cols].values
        
        X, y = [], []
        for i in range(self.seq_length, len(data)):
            X.append(features_scaled[i-self.seq_length:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)


class PatternRecognitionCNN:
    """CNN model for pattern recognition in price charts"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.model = None
        self.scaler = MinMaxScaler()
        
    def build_model(self, n_features: int, n_classes: int) -> Model:
        """Build CNN architecture for pattern recognition"""
        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', 
                   input_shape=(self.window_size, n_features)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Conv1D(32, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Conv1D(16, kernel_size=3, activation='relu'),
            BatchNormalization(),
            
            GlobalMaxPooling1D(),
            
            Dense(50, activation='relu'),
            Dropout(0.5),
            Dense(25, activation='relu'),
            Dropout(0.3),
            Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model


class EnsemblePredictor:
    """Ensemble model combining multiple prediction models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = {}
        self.weights = {}
        self.scalers = {}
        
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.weights[name] = weight
        
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series, 
                      X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train ensemble of models"""
        results = {}
        
        # Train LSTM
        if 'lstm' in self.config.ML_PARAMS['ensemble_models']:
            logger.info("Training LSTM model...")
            lstm_model = StockMarketLSTM(
                lookback_window=self.config.ML_PARAMS['lstm_lookback']
            )
            
            # Prepare LSTM data
            X_lstm, y_lstm = lstm_model.prepare_sequences(
                pd.concat([X_train, y_train], axis=1), [y_train.name]
            )
            
            # Split for validation
            split_idx = int(len(X_lstm) * 0.8)
            X_lstm_train, X_lstm_val = X_lstm[:split_idx], X_lstm[split_idx:]
            y_lstm_train, y_lstm_val = y_lstm[:split_idx], y_lstm[split_idx:]
            
            history = lstm_model.train(X_lstm_train, y_lstm_train, X_lstm_val, y_lstm_val)
            self.models['lstm'] = lstm_model
            results['lstm'] = history
        
        # Train XGBoost
        if 'xgboost' in self.config.ML_PARAMS['ensemble_models']:
            logger.info("Training XGBoost model...")
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config.ML_PARAMS['random_state']
            )
            xgb_model.fit(X_train, y_train)
            self.models['xgboost'] = xgb_model
            
            # Evaluate
            val_pred = xgb_model.predict(X_val)
            val_score = mean_squared_error(y_val, val_pred)
            results['xgboost'] = {'val_loss': val_score}
        
        # Train LightGBM
        if 'lightgbm' in self.config.ML_PARAMS['ensemble_models']:
            logger.info("Training LightGBM model...")
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config.ML_PARAMS['random_state']
            )
            lgb_model.fit(X_train, y_train)
            self.models['lightgbm'] = lgb_model
            
            # Evaluate
            val_pred = lgb_model.predict(X_val)
            val_score = mean_squared_error(y_val, val_pred)
            results['lightgbm'] = {'val_loss': val_score}
        
        # Calculate ensemble weights based on validation performance
        self._calculate_ensemble_weights(X_val, y_val)
        
        return results
    
    def _calculate_ensemble_weights(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Calculate optimal ensemble weights based on validation performance"""
        errors = {}
        
        for name, model in self.models.items():
            try:
                if name == 'lstm':
                    # Special handling for LSTM
                    X_lstm, _ = model.prepare_sequences(
                        pd.concat([X_val, y_val], axis=1), [y_val.name]
                    )
                    if len(X_lstm) > 0:
                        pred = model.predict(X_lstm)
                        if len(pred) > 0:
                            # Align predictions with actual values
                            aligned_actual = y_val.iloc[-len(pred):].values
                            if len(aligned_actual) == len(pred.flatten()):
                                errors[name] = mean_squared_error(aligned_actual, pred.flatten())
                            else:
                                errors[name] = float('inf')
                        else:
                            errors[name] = float('inf')
                    else:
                        errors[name] = float('inf')
                else:
                    pred = model.predict(X_val)
                    errors[name] = mean_squared_error(y_val, pred)
            except Exception as e:
                logger.warning(f"Error calculating weight for {name}: {str(e)}")
                errors[name] = float('inf')
        
        # Convert errors to weights (inverse relationship)
        total_inv_error = sum(1 / max(error, 1e-8) for error in errors.values())
        
        for name, error in errors.items():
            self.weights[name] = (1 / max(error, 1e-8)) / total_inv_error
        
        logger.info(f"Ensemble weights: {self.weights}")
    
    def predict(self, X: pd.DataFrame, target_col: str = None) -> np.ndarray:
        """Make ensemble predictions"""
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if name == 'lstm' and target_col:
                    # Special handling for LSTM
                    X_lstm, _ = model.prepare_sequences(
                        pd.concat([X, pd.Series([0]*len(X), name=target_col)], axis=1), 
                        [target_col]
                    )
                    if len(X_lstm) > 0:
                        pred = model.predict(X_lstm)
                        predictions[name] = pred.flatten()
                    else:
                        predictions[name] = np.array([])
                else:
                    pred = model.predict(X)
                    predictions[name] = pred
                    
            except Exception as e:
                logger.warning(f"Error in prediction for {name}: {str(e)}")
                continue
        
        if not predictions:
            raise ValueError("No models could make predictions")
        
        # Find common length
        min_length = min(len(pred) for pred in predictions.values() if len(pred) > 0)
        
        if min_length == 0:
            raise ValueError("All predictions are empty")
        
        # Ensemble prediction
        ensemble_pred = np.zeros(min_length)
        total_weight = 0
        
        for name, pred in predictions.items():
            if len(pred) >= min_length:
                weight = self.weights.get(name, 1.0)
                ensemble_pred += weight * pred[-min_length:]
                total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return ensemble_pred
    
    def save_models(self, directory: str):
        """Save all models in the ensemble"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            if name == 'lstm':
                model.model.save(f"{directory}/lstm_model.h5")
                joblib.dump(model.scaler, f"{directory}/lstm_scaler.pkl")
                joblib.dump(model.feature_scaler, f"{directory}/lstm_feature_scaler.pkl")
            else:
                joblib.dump(model, f"{directory}/{name}_model.pkl")
        
        # Save weights
        joblib.dump(self.weights, f"{directory}/ensemble_weights.pkl")
    
    def load_models(self, directory: str):
        """Load all models in the ensemble"""
        import os
        
        # Load weights
        weights_path = f"{directory}/ensemble_weights.pkl"
        if os.path.exists(weights_path):
            self.weights = joblib.load(weights_path)
        
        # Load LSTM
        lstm_path = f"{directory}/lstm_model.h5"
        if os.path.exists(lstm_path):
            lstm_model = StockMarketLSTM()
            lstm_model.model = tf.keras.models.load_model(lstm_path)
            lstm_model.scaler = joblib.load(f"{directory}/lstm_scaler.pkl")
            lstm_model.feature_scaler = joblib.load(f"{directory}/lstm_feature_scaler.pkl")
            self.models['lstm'] = lstm_model
        
        # Load other models
        for model_name in ['xgboost', 'lightgbm']:
            model_path = f"{directory}/{model_name}_model.pkl"
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)


class AnomalyDetector:
    """Autoencoder-based anomaly detection for unusual market patterns"""
    
    def __init__(self, encoding_dim: int = 32):
        self.encoding_dim = encoding_dim
        self.autoencoder = None
        self.encoder = None
        self.scaler = StandardScaler()
        self.threshold = None
        
    def build_autoencoder(self, input_dim: int) -> Model:
        """Build autoencoder architecture"""
        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(128, activation='relu')(input_layer)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='linear')(decoded)
        
        # Models
        self.autoencoder = Model(input_layer, decoded)
        self.encoder = Model(input_layer, encoded)
        
        self.autoencoder.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'
        )
        
        return self.autoencoder
    
    def train(self, X_train: np.ndarray, validation_split: float = 0.2) -> Dict:
        """Train the autoencoder"""
        if self.autoencoder is None:
            self.build_autoencoder(X_train.shape[1])
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X_train)
        
        history = self.autoencoder.fit(
            X_scaled, X_scaled,
            epochs=100,
            batch_size=32,
            validation_split=validation_split,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True)
            ],
            verbose=1
        )
        
        # Calculate threshold for anomaly detection
        predictions = self.autoencoder.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - predictions, 2), axis=1)
        self.threshold = np.percentile(mse, 95)  # 95th percentile as threshold
        
        return history.history
    
    def detect_anomalies(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies in new data"""
        if self.autoencoder is None or self.threshold is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.autoencoder.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - predictions, 2), axis=1)
        
        anomalies = mse > self.threshold
        anomaly_scores = mse
        
        return anomalies, anomaly_scores


class MarketRegimeDetector:
    """Hidden Markov Model for detecting market regime changes"""
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = None
        self.regime_names = ['Bull Market', 'Bear Market', 'Sideways Market']
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for regime detection"""
        features = []
        
        # Returns
        returns = data['Close'].pct_change().dropna()
        features.append(returns.values)
        
        # Volatility
        volatility = returns.rolling(20).std()
        features.append(volatility.dropna().values)
        
        # Volume changes
        volume_change = data['Volume'].pct_change().dropna()
        features.append(volume_change.values[1:])  # Align with volatility
        
        # Combine features
        min_length = min(len(f) for f in features)
        feature_matrix = np.column_stack([f[-min_length:] for f in features])
        
        return feature_matrix
    
    def detect_regimes(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect market regimes using Gaussian Mixture Model as proxy for HMM"""
        from sklearn.mixture import GaussianMixture
        
        # Use GMM as a simpler alternative to HMM
        gmm = GaussianMixture(n_components=self.n_regimes, random_state=42)
        regimes = gmm.fit_predict(features)
        probabilities = gmm.predict_proba(features)
        
        self.model = gmm
        return regimes, probabilities
    
    def get_current_regime(self, recent_features: np.ndarray) -> Tuple[int, str, float]:
        """Get current market regime"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        regime = self.model.predict(recent_features.reshape(1, -1))[0]
        probabilities = self.model.predict_proba(recent_features.reshape(1, -1))[0]
        confidence = probabilities[regime]
        
        regime_name = self.regime_names[regime] if regime < len(self.regime_names) else f"Regime {regime}"
        
        return regime, regime_name, confidence