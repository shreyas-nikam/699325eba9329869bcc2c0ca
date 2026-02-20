import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Lambda
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import requests
import os

# Force yfinance to use requests instead of curl_cffi to avoid chrome impersonation issues
yf._CURL_INSTALLED = False

# Suppress warnings for cleaner output globally (can be controlled by setup_environment)
warnings.filterwarnings('ignore')

# --- Utility Functions and Configuration ---


def setup_environment(seed=42):
    """Sets random seeds for reproducibility and suppresses warnings."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    print(
        f"Environment setup: Random seed set to {seed}, warnings suppressed.")

# Reparameterization trick for VAE - Keep as a global helper or nested if only used once


def sampling(args):
    """
    Implements the reparameterization trick to sample from the latent space.
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(tf.keras.Model):
    """Variational Autoencoder custom Keras model."""

    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        # data is typically (x, y) or x. For VAE, x is also y.
        x_input = data[0] if isinstance(data, (list, tuple)) else data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x_input)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.keras.ops.mean(
                tf.keras.ops.square(x_input - reconstruction), axis=-1
            )
            kl_loss = -0.5 * tf.keras.ops.mean(
                1 + z_log_var -
                tf.keras.ops.square(z_mean) - tf.keras.ops.exp(z_log_var),
                axis=-1,
            )
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs):
        # inputs could be (x, y) if called directly by model.fit or model.predict
        x_input = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        z_mean, z_log_var, z = self.encoder(x_input)
        reconstruction = self.decoder(z)
        return reconstruction


def acquire_and_preprocess_market_data(start_date='2008-01-01', end_date='2024-12-31'):
    """
    Acquires multi-asset market data from Yahoo Finance, calculates returns,
    and engineers cross-sectional features.
    """
    # load it from market_returns.csv and known_anomaly_dates.csv
    market_returns = pd.read_csv(
        'market_returns.csv', index_col=0, parse_dates=True)
    known_anomaly_dates = pd.read_csv(
        'known_anomaly_dates.csv', parse_dates=['known_anomaly_dates'])['known_anomaly_dates'].tolist()
    return market_returns, known_anomaly_dates


def prepare_data_for_ae(returns_df, train_end_date='2023-01-01', feature_cols=None):
    """
    Scales the market data and splits it into training and testing sets.
    The training set aims to represent 'normal' market behavior.
    """
    if feature_cols is None:
        # Exclude 'VIX' from features used by AE if VIX is intended for coloring latent space,
        # or if it's considered an outcome rather than an input for normal market dynamics.
        # For this lab, let's keep VIX in for the AE to learn its relation to other assets,
        # but the original text suggested excluding cross_corr for AE, I'll follow that.
        feature_cols = [col for col in returns_df.columns if col not in [
            'cross_corr', 'cross_vol']]

    X = returns_df[feature_cols]

    # Standardize features to zero mean and unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

    # Split data into training and test sets
    train_mask = X_scaled_df.index < train_end_date
    test_mask = X_scaled_df.index >= train_end_date

    X_train = X_scaled_df[train_mask]
    X_test = X_scaled_df[test_mask]

    print(
        f"\nTraining set: {X_train.shape[0]} days, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} days")
    print(f"Features used for Autoencoder: {feature_cols}")

    return X_train, X_test, scaler, train_mask, test_mask, feature_cols

# --- Autoencoder Model Definition and Training ---


def build_autoencoder(input_dim, encoding_dim=4, hidden_dims=[32, 16], dropout_rate=0.2):
    """
    Constructs a symmetric dense autoencoder for anomaly detection.
    """
    inputs = Input(shape=(input_dim,), name='input_layer')
    x = inputs

    # Encoder
    for i, dim in enumerate(hidden_dims):
        x = Dense(dim, activation='relu', name=f'encoder_hidden_{i+1}')(x)
        x = BatchNormalization(name=f'encoder_bn_{i+1}')(x)
        x = Dropout(dropout_rate, name=f'encoder_dropout_{i+1}')(x)

    # Bottleneck (latent representation)
    latent = Dense(encoding_dim, activation='relu', name='bottleneck_layer')(x)

    # Decoder (mirror of encoder)
    x = latent
    for i, dim in enumerate(reversed(hidden_dims)):
        x = Dense(dim, activation='relu', name=f'decoder_hidden_{i+1}')(x)
        x = BatchNormalization(name=f'decoder_bn_{i+1}')(x)
        x = Dropout(dropout_rate, name=f'decoder_dropout_{i+1}')(x)

    # Output layer (reconstruct input)
    outputs = Dense(input_dim, activation='linear', name='output_layer')(x)

    # Full autoencoder model
    autoencoder = Model(inputs, outputs, name='autoencoder')

    # Encoder-only model for latent space visualization
    encoder = Model(inputs, latent, name='encoder')

    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder


def train_autoencoder(model, X_train_data, epochs=100, batch_size=64, validation_split=0.15, patience=10, verbose=0):
    """
    Trains the autoencoder model on the provided training data.
    """
    print(f"\nTraining autoencoder on {X_train_data.shape[0]} samples...")
    # Callbacks for early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=patience, restore_best_weights=True)

    history = model.fit(
        X_train_data, X_train_data,  # Input = Target (self-reconstruction)
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        shuffle=True,
        verbose=verbose  # Suppress verbose output for cleaner notebook, tqdm will provide progress
    )
    print("Autoencoder training complete.")
    return history

# --- Anomaly Detection and Analysis ---


def calculate_reconstruction_errors(model, X_data):
    """
    Predicts reconstructions and calculates the reconstruction error (MSE) for each sample.
    """
    X_pred = model.predict(X_data, verbose=0)
    # Calculate MSE for each sample across all features
    errors = np.mean(np.square(X_data - X_pred), axis=1)
    return errors


def set_anomaly_thresholds(train_errors, percentiles=[95, 99]):
    """
    Calculates percentile-based anomaly thresholds from training reconstruction errors.
    """
    thresholds = {p: np.percentile(train_errors, p) for p in percentiles}
    return thresholds


def summarize_anomaly_results(market_returns_df, test_mask, test_recon_errors, anomaly_thresholds):
    """
    Flags anomalies in the test set and creates a summary DataFrame.
    """
    test_dates = market_returns_df.index[test_mask]
    anomaly_flags_99 = test_recon_errors > anomaly_thresholds[99]
    anomaly_flags_95 = test_recon_errors > anomaly_thresholds[95]

    n_anomalies_99 = anomaly_flags_99.sum()
    n_anomalies_95 = anomaly_flags_95.sum()

    print(
        f"\nAnomalies detected (99th percentile): {n_anomalies_99} / {len(test_recon_errors)} ({n_anomalies_99 / len(test_recon_errors):.1%})")
    print(
        f"Anomalies detected (95th percentile): {n_anomalies_95} / {len(test_recon_errors)} ({n_anomalies_95 / len(test_recon_errors):.1%})")

    anomaly_results_df = pd.DataFrame({
        'date': test_dates,
        'recon_error': test_recon_errors,
        'SP500_ret': market_returns_df.loc[test_mask, 'SP500'].values,
        'VIX': market_returns_df.loc[test_mask, 'VIX'].values,
        'anomaly_flag_99': anomaly_flags_99,
        'anomaly_flag_95': anomaly_flags_95
    }).set_index('date')

    # Show top 10 anomalous days by reconstruction error
    top_anomalies_df = anomaly_results_df.nlargest(10, 'recon_error')
    print("\nTop 10 anomalous days by reconstruction error (99th percentile threshold is for visualization):")
    print(top_anomalies_df[['recon_error', 'SP500_ret',
          'VIX', 'anomaly_flag_99']].to_string())

    return anomaly_results_df


def analyze_per_feature_errors(model, X_data_scaled_df, anomaly_dates_to_analyze, feature_names):
    """
    Calculates per-feature reconstruction errors for selected anomalous dates.
    """
    print(
        f"\nAnalyzing per-feature errors for {len(anomaly_dates_to_analyze)} anomalous days...")

    per_feature_errors = pd.DataFrame(
        index=anomaly_dates_to_analyze, columns=feature_names)
    X_pred_all = model.predict(X_data_scaled_df, verbose=0)
    X_pred_df = pd.DataFrame(
        X_pred_all, index=X_data_scaled_df.index, columns=feature_names)

    for date in anomaly_dates_to_analyze:
        if date not in X_data_scaled_df.index:
            print(
                f"Warning: Date {date.strftime('%Y-%m-%d')} not found in scaled data. Skipping.")
            continue

        original_scaled_features = X_data_scaled_df.loc[date].values
        reconstructed_scaled_features = X_pred_df.loc[date].values
        feature_squared_errors = np.square(
            original_scaled_features - reconstructed_scaled_features)
        per_feature_errors.loc[date] = feature_squared_errors
    return per_feature_errors, X_pred_df

# --- Visualization Functions ---


def plot_training_history(history):
    """
    Plots the training and validation loss curves.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Training: Reconstruction Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_error_distribution(train_errors, thresholds):
    """
    Plots a histogram of training reconstruction errors with anomaly thresholds.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(train_errors, bins=50, kde=True, stat='density',
                 alpha=0.7, color='skyblue', label='Train Errors')

    for p, t in thresholds.items():
        plt.axvline(x=t, color='red' if p == 99 else 'orange',
                    linestyle='--', label=f'{p}th Percentile ({t:.4f})')

    plt.title(
        'Distribution of Training Reconstruction Errors and Anomaly Thresholds')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Log scale helps visualize the heavy tail of errors
    plt.show()


def plot_anomaly_time_series(anomaly_df, thresholds, known_dates):
    """
    Generates a dual-panel plot showing S&P 500 returns, reconstruction errors,
    anomaly thresholds, and flagged dates.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Top panel: S&P 500 returns with flagged anomalies
    ax1.plot(anomaly_df.index, anomaly_df['SP500_ret'],
             color='black', linewidth=0.7, label='S&P 500 Daily Return')

    # Highlight known anomaly dates
    known_anom_in_test = anomaly_df.index.intersection(known_dates)
    if not known_anom_in_test.empty:
        ax1.scatter(known_anom_in_test, anomaly_df.loc[known_anom_in_test, 'SP500_ret'],
                    marker='o', color='purple', s=70, zorder=6, label='Known Market Event')

    # Highlight model-flagged anomalies (99th percentile)
    model_anom_dates = anomaly_df[anomaly_df['flag_99pctl']].index
    if not model_anom_dates.empty:
        ax1.scatter(model_anom_dates, anomaly_df.loc[model_anom_dates, 'SP500_ret'],
                    c='red', s=50, zorder=5, label='AE Anomaly (99%)')

    ax1.set_ylabel('S&P 500 Daily Return')
    ax1.set_title('Market Returns and Autoencoder Anomalies')
    ax1.legend()
    ax1.grid(True)

    # Bottom panel: Reconstruction error with thresholds
    ax2.plot(anomaly_df.index, anomaly_df['recon_error'], color='blue',
             linewidth=0.7, alpha=0.7, label='Reconstruction Error (MSE)')

    # Add anomaly thresholds
    ax2.axhline(y=thresholds[99], color='red', linestyle='--',
                label=f'99th Pct Threshold ({thresholds[99]:.4f})')
    ax2.axhline(y=thresholds[95], color='orange', linestyle='--',
                label=f'95th Pct Threshold ({thresholds[95]:.4f})')

    # Highlight area above 99th percentile threshold
    ax2.fill_between(anomaly_df.index, thresholds[99], anomaly_df['recon_error'],
                     where=(anomaly_df['recon_error'] > thresholds[99]), color='red', alpha=0.2)

    ax2.set_xlabel('Date')
    ax2.set_ylabel('Reconstruction Error (MSE)')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(
        'Autoencoder Anomaly Detection: Market Returns & Reconstruction Error', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def plot_per_feature_errors(per_feature_errors_df, X_data_scaled_df, X_pred_df, anomaly_dates_to_analyze, feature_names):
    """
    Visualizes per-feature reconstruction errors as an "anomaly gallery" and heatmap.
    """
    if per_feature_errors_df.empty or anomaly_dates_to_analyze.empty:
        print("No per-feature errors to plot or no anomalous dates provided.")
        return

    # Anomaly Gallery: Original vs. Reconstructed
    fig, axes = plt.subplots(len(anomaly_dates_to_analyze), 2, figsize=(
        18, 5 * len(anomaly_dates_to_analyze)))
    if len(anomaly_dates_to_analyze) == 1:  # Handle single row subplot case
        axes = [axes]

    for i, date in enumerate(anomaly_dates_to_analyze):
        if date not in X_data_scaled_df.index:
            continue  # Already warned in analyze_per_feature_errors

        original_scaled_features = X_data_scaled_df.loc[date].values
        reconstructed_scaled_features = X_pred_df.loc[date].values
        # Use already calculated errors
        feature_squared_errors = per_feature_errors_df.loc[date].values

        df_plot = pd.DataFrame({
            'Original': original_scaled_features,
            'Reconstructed': reconstructed_scaled_features
        }, index=feature_names)

        ax = axes[i][0] if len(anomaly_dates_to_analyze) > 1 else axes[0][0]
        df_plot.plot(kind='bar', ax=ax, alpha=0.7)
        ax.set_title(
            f'Original vs. Reconstructed Features (Scaled) on {date.strftime("%Y-%m-%d")}')
        ax.set_ylabel('Scaled Value')
        ax.tick_params(axis='x', rotation=45)

        # Highlight features with highest error
        top_error_features = df_plot['Original'].index[np.argsort(
            feature_squared_errors)[-3:]]
        colors = [
            'red' if f in top_error_features else 'blue' for f in df_plot.index]
        for bar, color in zip(ax.patches, colors):
            bar.set_color(color)

    plt.tight_layout()
    plt.show()

    # Per-feature error heatmap
    plt.figure(figsize=(14, 0.8 * len(anomaly_dates_to_analyze) + 2))
    sns.heatmap(per_feature_errors_df.astype(float), cmap='YlOrRd',
                annot=True, fmt=".2f", linewidths=.5, linecolor='lightgray')
    plt.title(
        'Per-Feature Reconstruction Error Heatmap for Anomalous Days (Squared Error)')
    plt.xlabel('Features')
    plt.ylabel('Date')
    plt.tight_layout()
    plt.show()


def build_and_visualize_latent_space(X_train_data, X_test_data, market_returns_df, train_mask_idx, test_mask_idx, ae_features, anomaly_results_df, encoding_dim=2):
    """
    Builds a special autoencoder with a 2D latent space for visualization,
    trains it, and plots the latent space colored by VIX Daily Change.
    """
    input_dim_2d = X_train_data.shape[1]

    # Build a separate autoencoder with encoding_dim=2 for visualization
    ae_2d, enc_2d = build_autoencoder(
        input_dim_2d, encoding_dim=encoding_dim, hidden_dims=[32, 16])

    print(
        f"\nTraining 2D latent space autoencoder (encoding_dim={encoding_dim})...")
    # Train the 2D AE (can use fewer epochs as it's mainly for visualization)
    train_autoencoder(ae_2d, X_train_data, epochs=50, batch_size=64,
                      validation_split=0.15, patience=10, verbose=0)
    print("2D Autoencoder training complete.")

    # Get latent space representations for training data
    Z_train = enc_2d.predict(X_train_data, verbose=0)
    Z_test = enc_2d.predict(X_test_data, verbose=0)

    # Prepare VIX data for coloring
    vix_change = market_returns_df['VIX'].pct_change().dropna()

    # Align VIX data with Z_train/Z_test indices
    Z_train_df = pd.DataFrame(Z_train, index=X_train_data.index)
    Z_test_df = pd.DataFrame(Z_test, index=X_test_data.index)

    # Ensure VIX values align with the latent space data
    # Drop dates where VIX data is not available after pct_change()
    vix_train_aligned = vix_change.reindex(Z_train_df.index).fillna(
        0)  # Fillna with 0 for missing VIX changes
    vix_test_aligned = vix_change.reindex(Z_test_df.index).fillna(0)

    # Plot latent space for training data
    plt.figure(figsize=(12, 10))
    scatter_train = plt.scatter(Z_train_df.iloc[:, 0], Z_train_df.iloc[:, 1],
                                c=vix_train_aligned, cmap='RdYlGn_r', alpha=0.5, s=15, label='Training Data')
    plt.colorbar(scatter_train, label='VIX Daily Change (%)')

    # Plot latent space for test data, highlighting anomalies
    test_anomalies_2d = anomaly_results_df[anomaly_results_df['flag_99pctl']].index.intersection(
        Z_test_df.index)

    # Only scatter test data points that are not anomalies
    normal_test_indices = Z_test_df.index.difference(test_anomalies_2d)

    if not normal_test_indices.empty:
        plt.scatter(Z_test_df.loc[normal_test_indices, 0], Z_test_df.loc[normal_test_indices, 1],
                    c=vix_test_aligned.loc[normal_test_indices], cmap='RdYlGn_r',
                    marker='x', alpha=0.3, s=20, label='Test Data (Normal)')

    if not test_anomalies_2d.empty:
        plt.scatter(Z_test_df.loc[test_anomalies_2d, 0], Z_test_df.loc[test_anomalies_2d, 1],
                    color='red', marker='^', s=100, zorder=5, label='Test Data (Anomaly 99%)')

    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Autoencoder Latent Space (Colored by VIX Daily Change)')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Comparison with Other Models ---


def compare_anomaly_detectors(X_train_data, X_test_data, anomaly_df, known_dates_list, ae_test_errors):
    """
    Compares the Autoencoder's performance against Isolation Forest and Mahalanobis distance.
    Uses known anomaly dates as sparse ground truth for evaluation metrics.
    """
    print("\n--- Comparing Anomaly Detection Models ---")

    # 1. Isolation Forest
    print("Running Isolation Forest...")
    # Contamination based on expected anomaly rate
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_forest.fit(X_train_data)
    # Higher score = more anomalous
    iso_scores = -iso_forest.decision_function(X_test_data)

    # 2. Mahalanobis Distance
    print("Calculating Mahalanobis Distance...")
    # Calculate inverse covariance matrix from training data
    cov_inv = np.linalg.pinv(np.cov(X_train_data, rowvar=False))
    mean_train = np.mean(X_train_data, axis=0)

    mahal_scores = np.array([distance.mahalanobis(
        x, mean_train, cov_inv) for x in X_test_data.values])

    # Prepare ground truth labels (1 for known anomaly, 0 otherwise)
    # This is a sparse ground truth for market data
    test_dates = anomaly_df.index
    y_true_labels = np.array(
        [1 if d in known_dates_list else 0 for d in test_dates]).astype(int)

    # Store model scores
    model_scores = {
        'Autoencoder': ae_test_errors,
        'Isolation Forest': iso_scores,
        'Mahalanobis': mahal_scores
    }

    print("\nEvaluation Metrics (using known market events as ground truth):")
    if y_true_labels.sum() > 0:  # Only evaluate if there are known anomalies in the test set
        for name, scores in model_scores.items():
            auc = roc_auc_score(y_true_labels, scores)
            ap = average_precision_score(y_true_labels, scores)
            print(f"{name:20s}: ROC AUC={auc:.3f}, Avg Precision={ap:.3f}")

        # Visualize Precision-Recall Curves if meaningful (enough positive samples)
        plt.figure(figsize=(10, 7))
        for name, scores in model_scores.items():
            precision, recall, _ = precision_recall_curve(
                y_true_labels, scores)
            plt.plot(recall, precision,
                     label=f'{name} (AP={average_precision_score(y_true_labels, scores):.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve for Anomaly Detection Baselines')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No known anomaly dates found in the test set for quantitative evaluation.")
        print("Qualitative comparison will rely on visual inspection of flagged dates.")

# --- Variational Autoencoder (VAE) Flow ---


def build_vae_components(input_dim, latent_dim=4, hidden_dims=[32, 16], dropout_rate=0.2):
    """
    Constructs the encoder and decoder components for a Variational Autoencoder.
    """
    # Encoder
    encoder_inputs = Input(shape=(input_dim,), name='encoder_input_vae')
    x = encoder_inputs
    for i, dim in enumerate(hidden_dims):
        x = Dense(dim, activation='relu', name=f'vae_encoder_hidden_{i+1}')(x)
        x = BatchNormalization(name=f'vae_encoder_bn_{i+1}')(x)
        x = Dropout(dropout_rate, name=f'vae_encoder_dropout_{i+1}')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,),
               name='z_sampling')([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='vae_encoder')

    # Decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling_input')
    x = latent_inputs
    for i, dim in enumerate(reversed(hidden_dims)):
        x = Dense(dim, activation='relu', name=f'vae_decoder_hidden_{i+1}')(x)
        x = BatchNormalization(name=f'vae_decoder_bn_{i+1}')(x)
        x = Dropout(dropout_rate, name=f'vae_decoder_dropout_{i+1}')(x)
    decoder_outputs = Dense(
        input_dim, activation='linear', name='vae_output_layer')(x)
    decoder = Model(latent_inputs, decoder_outputs, name='vae_decoder')

    return encoder, decoder


def train_and_evaluate_vae(X_train_data, X_test_data, input_dim, latent_dim=4, hidden_dims=[32, 16], dropout_rate=0.2,
                           epochs=50, batch_size=64, validation_split=0.15, patience=10, verbose=0):
    """
    Builds, trains, and evaluates a Variational Autoencoder.
    """
    print("\n--- Variational Autoencoder (VAE) Anomaly Detection ---")
    vae_encoder, vae_decoder = build_vae_components(
        input_dim, latent_dim, hidden_dims, dropout_rate)
    vae_model = VAE(vae_encoder, vae_decoder)
    # Loss is handled within the custom train_step
    vae_model.compile(optimizer='adam')

    print("\nVAE Model Summary:")
    vae_model.build((None, input_dim))  # Build the model to print summary
    vae_model.summary()

    print("\nTraining VAE...")
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=patience, restore_best_weights=True)
    vae_history = vae_model.fit(X_train_data, X_train_data, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose,
                                callbacks=[early_stopping])
    print("VAE trained successfully.")

    X_test_reconstructed_vae = vae_model.predict(X_test_data, verbose=0)

    print("\nOriginal Test Data (first 5 rows):")
    print(X_test_data.head())

    print("\nReconstructed Test Data (first 5 rows from VAE):")
    print(pd.DataFrame(X_test_reconstructed_vae,
          index=X_test_data.index, columns=X_test_data.columns).head())

    return vae_model, vae_history, X_test_reconstructed_vae

# --- Main Orchestration Function ---


def run_market_anomaly_detection_pipeline(
    start_date='2008-01-01',
    end_date='2024-12-31',
    train_end_date='2023-01-01',
    ae_encoding_dim=4,
    ae_hidden_dims=[32, 16],
    ae_dropout_rate=0.2,
    ae_epochs=100,
    ae_batch_size=64,
    ae_validation_split=0.15,
    ae_patience=10,
    visualize_2d_latent_space=True,
    vae_latent_dim=4,
    vae_epochs=50,
    top_n_anomalies_for_detail=5,
    plot_verbose=True,  # Control if plots are shown using plt.show()
    # Keras fit verbose level (0=silent, 1=progress bar, 2=one line per epoch)
    ae_train_verbose=0,
    vae_train_verbose=0,  # Keras fit verbose level for VAE
    seed=42
):
    """
    Runs the complete market anomaly detection pipeline using Autoencoders and VAEs.

    Args:
        start_date (str): Start date for data acquisition.
        end_date (str): End date for data acquisition.
        train_end_date (str): End date for the training period.
        ae_encoding_dim (int): Dimension of the Autoencoder's latent space.
        ae_hidden_dims (list): Hidden layer dimensions for AE encoder/decoder.
        ae_dropout_rate (float): Dropout rate for AE.
        ae_epochs (int): Maximum number of epochs for AE training.
        ae_batch_size (int): Batch size for AE training.
        ae_validation_split (float): Fraction of the training data to be used as validation data for AE.
        ae_patience (int): Number of epochs with no improvement after which AE training will be stopped.
        visualize_2d_latent_space (bool): If True, a separate 2D AE will be built and its latent space visualized.
        vae_latent_dim (int): Dimension of the VAE's latent space.
        vae_epochs (int): Maximum number of epochs for VAE training.
        top_n_anomalies_for_detail (int): Number of top anomalous days to analyze in detail per feature.
        plot_verbose (bool): If True, plots will be displayed using `plt.show()`. If False, plots are generated but not displayed.
        ae_train_verbose (int): Verbosity mode for Autoencoder training (0, 1, or 2).
        vae_train_verbose (int): Verbosity mode for Variational Autoencoder training (0, 1, or 2).
        seed (int): Random seed for reproducibility across the entire pipeline.

    Returns:
        dict: A dictionary containing key results, models, and dataframes from the pipeline.
    """
    setup_environment(seed)

    # 1. Acquire and Preprocess Data
    market_returns, known_anomaly_dates = acquire_and_preprocess_market_data(
        start_date, end_date)

    # 2. Prepare Data for Autoencoder
    X_train, X_test, scaler, train_mask, test_mask, ae_features = prepare_data_for_ae(
        market_returns, train_end_date
    )
    input_dim = X_train.shape[1]

    # 3. Build and Train Autoencoder
    print(
        f"\nBuilding Autoencoder with {ae_encoding_dim}-dimensional latent space...")
    autoencoder, encoder = build_autoencoder(
        input_dim, encoding_dim=ae_encoding_dim, hidden_dims=ae_hidden_dims, dropout_rate=ae_dropout_rate
    )
    print("\nAutoencoder Model Summary:")
    autoencoder.summary()
    ae_training_history = train_autoencoder(
        autoencoder, X_train,
        epochs=ae_epochs, batch_size=ae_batch_size, validation_split=ae_validation_split, patience=ae_patience, verbose=ae_train_verbose
    )

    if plot_verbose:
        plot_training_history(ae_training_history)

    # 4. Calculate Reconstruction Errors and Set Thresholds
    train_recon_errors = calculate_reconstruction_errors(autoencoder, X_train)
    test_recon_errors = calculate_reconstruction_errors(autoencoder, X_test)
    anomaly_thresholds = set_anomaly_thresholds(train_recon_errors)

    print(
        f"\nTraining error stats: Mean={np.mean(train_recon_errors):.4f}, Std={np.std(train_recon_errors):.4f}")
    for p, t in anomaly_thresholds.items():
        print(f"Threshold ({p}th percentile): {t:.4f}")

    # 5. Summarize and Plot Anomalies
    anomaly_results_df = summarize_anomaly_results(
        market_returns, test_mask, test_recon_errors, anomaly_thresholds)

    if plot_verbose:
        plot_error_distribution(train_recon_errors, anomaly_thresholds)
        plot_anomaly_time_series(
            anomaly_results_df, anomaly_thresholds, known_anomaly_dates)

    # 6. Analyze Per-Feature Errors for Top Anomalies
    top_anomaly_dates = anomaly_results_df.nlargest(
        top_n_anomalies_for_detail, 'recon_error').index
    per_feature_errors_df, X_test_pred_df = analyze_per_feature_errors(
        autoencoder, X_test, top_anomaly_dates, ae_features
    )
    if plot_verbose:
        plot_per_feature_errors(
            per_feature_errors_df, X_test, X_test_pred_df, top_anomaly_dates, ae_features)

    # 7. Latent Space Visualization
    if visualize_2d_latent_space:
        build_and_visualize_latent_space(
            X_train, X_test, market_returns, train_mask, test_mask, ae_features, anomaly_results_df, encoding_dim=2
        )

    # 8. Compare with Other Anomaly Detectors
    compare_anomaly_detectors(
        X_train, X_test, anomaly_results_df, known_anomaly_dates, test_recon_errors)

    # 9. Variational Autoencoder (VAE) Pipeline
    vae_model, vae_history, X_test_reconstructed_vae = train_and_evaluate_vae(
        X_train, X_test, input_dim,
        latent_dim=vae_latent_dim, hidden_dims=ae_hidden_dims, dropout_rate=ae_dropout_rate,
        epochs=vae_epochs, batch_size=ae_batch_size, validation_split=ae_validation_split, patience=ae_patience, verbose=vae_train_verbose
    )
    if plot_verbose:
        # Plot VAE training history if desired, similar to AE
        # Re-use for VAE history too, works if history object has 'loss' and 'val_loss'
        plot_training_history(vae_history)

    results = {
        "market_returns": market_returns,
        "known_anomaly_dates": known_anomaly_dates,
        "X_train": X_train,
        "X_test": X_test,
        "scaler": scaler,
        "autoencoder": autoencoder,
        "encoder": encoder,
        "ae_training_history": ae_training_history,
        "train_recon_errors": train_recon_errors,
        "test_recon_errors": test_recon_errors,
        "anomaly_thresholds": anomaly_thresholds,
        "anomaly_results_df": anomaly_results_df,
        "per_feature_errors_df": per_feature_errors_df,
        "vae_model": vae_model,
        "vae_history": vae_history,
        "X_test_reconstructed_vae": X_test_reconstructed_vae
    }
    return results


if __name__ == "__main__":
    # Example usage:
    print("--- Starting Market Anomaly Detection Pipeline ---")
    pipeline_results = run_market_anomaly_detection_pipeline(
        start_date='2008-01-01',
        end_date='2024-12-31',
        train_end_date='2023-01-01',
        ae_encoding_dim=4,
        ae_epochs=100,
        vae_latent_dim=4,
        vae_epochs=50,
        top_n_anomalies_for_detail=5,
        plot_verbose=True,  # Set to False if you don't want plots to pop up
        ae_train_verbose=0,  # Set to 1 or 2 for progress bars during AE training
        vae_train_verbose=0,  # Set to 1 or 2 for progress bars during VAE training
        seed=42
    )

    print("\n--- Pipeline Execution Complete ---")
    print(
        "Access results using the 'pipeline_results' dictionary, e.g., pipeline_results['anomaly_results_df'].head()")
    print("\nExample: Top 5 Autoencoder Anomalies:")
    print(pipeline_results['anomaly_results_df'].nlargest(5, 'recon_error'))
