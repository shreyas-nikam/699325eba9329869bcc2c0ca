
# Streamlit Application Specification: Autoencoders for Market Anomaly Detection

## 1. Application Overview

### Purpose of the Application

The "Market Surveillance for Unusual Market Movements: An Autoencoder Approach" application serves as a sophisticated tool for Quant Analysts and Investment Professionals at Apex Asset Management. Its primary purpose is to **detect and analyze unusual market movements** across a multi-asset portfolio, such as flash crashes or sudden correlation breakdowns. By leveraging an Autoencoder (AE) model trained on "normal" market behavior, the application identifies anomalies through reconstruction error. It also provides an "Automated Market Anomaly Report," enables benchmarking against traditional methods like Isolation Forest and Mahalanobis distance, and offers insights into the nature of anomalies through per-feature error analysis and latent space visualization. This application aims to demonstrate a real-world workflow for proactive risk management and market surveillance, moving beyond reactive measures.

### High-Level Story Flow of the Application

The application guides the user (a Quant Analyst) through a structured workflow to set up and utilize an Autoencoder-based market surveillance system:

1.  **Data Acquisition & Preparation**: The analyst begins by selecting a historical date range to acquire multi-asset market data from Yahoo Finance. This data is then preprocessed, scaled, and split into training and testing sets, with a specific "normal" period designated for model training. Cross-sectional features are engineered to enrich the dataset.
2.  **Autoencoder Model & Training**: With the data prepared, the analyst proceeds to define the Autoencoder architecture. Key hyperparameters like the bottleneck dimension and hidden layer sizes are configured. The AE is then trained on the "normal" market data to learn underlying relationships, with visualization of the training progress.
3.  **Anomaly Detection & Thresholding**: Post-training, the model calculates reconstruction errors for both training and test data. The analyst sets anomaly thresholds (e.g., 95th or 99th percentile) based on the training error distribution. The application then flags potential anomalous days in the test set, displaying a time series of errors and market returns.
4.  **Anomaly Insights & Latent Space**: For flagged anomalous days, the analyst can perform a deep dive. This includes per-feature error analysis (anomaly gallery and heatmap) to understand *what* assets or features contributed most to the anomaly. Additionally, a 2D projection of the autoencoder's latent space is visualized, colored by market regime (e.g., VIX), to reveal the structural patterns learned by the model.
5.  **Model Benchmarking & VAE Extension**: To validate the Autoencoder's efficacy, its performance is compared against baseline anomaly detection methods (Isolation Forest, Mahalanobis distance) using Precision-Recall curves. The application also provides a brief conceptual introduction and demonstration of a Variational Autoencoder (VAE), highlighting its generative capabilities for synthetic data generation and stress testing, thus bridging to advanced financial modeling concepts.

This structured flow empowers the Quant Analyst to systematically identify, analyze, and contextualize unusual market movements, facilitating informed decision-making and robust risk management.

---

## 2. Code Requirements

### Imports

The `app.py` file must include the following import statements at the top:

```python
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt # For capturing plots
import seaborn as sns # Used by source.py plots
import tensorflow as tf # For EarlyStopping callback
from io import StringIO # For capturing model summary output

# Import all functions from source.py
from source import *
```

### `st.session_state` Design

All relevant data, models, and intermediate results will be stored in `st.session_state` to maintain state across user interactions and page navigations.

**Initialization (at the top of `app.py`):**

```python
# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Data Acquisition & Preparation"

# Initialize state variables for data acquisition and preparation
if 'market_returns' not in st.session_state:
    st.session_state.market_returns = None
if 'known_anomaly_dates' not in st.session_state:
    st.session_state.known_anomaly_dates = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'train_mask' not in st.session_state:
    st.session_state.train_mask = None
if 'test_mask' not in st.session_state:
    st.session_state.test_mask = None
if 'ae_features' not in st.session_state:
    st.session_state.ae_features = None
if 'input_dim' not in st.session_state:
    st.session_state.input_dim = None

# Initialize state variables for Autoencoder model and training
if 'autoencoder' not in st.session_state:
    st.session_state.autoencoder = None
if 'encoder' not in st.session_state:
    st.session_state.encoder = None
if 'model_summary' not in st.session_state: # To store the text summary of the AE model
    st.session_state.model_summary = None
if 'ae_training_history' not in st.session_state:
    st.session_state.ae_training_history = None

# Initialize state variables for anomaly detection and thresholding
if 'train_recon_errors' not in st.session_state:
    st.session_state.train_recon_errors = None
if 'test_recon_errors' not in st.session_state:
    st.session_state.test_recon_errors = None
if 'anomaly_thresholds' not in st.session_state:
    st.session_state.anomaly_thresholds = None
if 'anomaly_results_df' not in st.session_state:
    st.session_state.anomaly_results_df = None
if 'top_anomalies_df' not in st.session_state: # Stores top N anomalous days
    st.session_state.top_anomalies_df = None

# Initialize state variables for VAE extension
if 'vae_model' not in st.session_state:
    st.session_state.vae_model = None
if 'vae_encoder' not in st.session_state:
    st.session_state.vae_encoder = None
if 'vae_decoder' not in st.session_state:
    st.session_state.vae_decoder = None
if 'vae_model_summary' not in st.session_state: # To store the text summary of the VAE model
    st.session_state.vae_model_summary = None
if 'vae_trained_flag' not in st.session_state: # Flag to indicate if VAE has been trained
    st.session_state.vae_trained_flag = False
```

### Application Structure and Flow

The application simulates a multi-page experience using a `st.sidebar.selectbox`. A helper function `clear_matplotlib_cache()` will be defined and called at the beginning of each page render to manage plot lifecycle.

```python
# Helper function to clear matplotlib figures
def clear_matplotlib_cache():
    plt.close('all')

st.sidebar.title("Navigation")
page_selection = st.sidebar.selectbox(
    "Go to",
    [
        "Data Acquisition & Preparation",
        "Autoencoder Model & Training",
        "Anomaly Detection & Thresholding",
        "Anomaly Insights & Latent Space",
        "Model Benchmarking & VAE Extension"
    ]
)
st.session_state.current_page = page_selection

# --- Page: Data Acquisition & Preparation ---
if st.session_state.current_page == "Data Acquisition & Preparation":
    clear_matplotlib_cache()
    st.title("Market Surveillance for Unusual Market Movements: An Autoencoder Approach")

    # Markdown Section 1 & 2 Introduction
    st.markdown(f"## 1. Setting the Stage: Essential Tools for Market Surveillance")
    st.markdown(f"As a Quant Analyst at Apex Asset Management, my first step in setting up our market surveillance system is to ensure I have all the necessary Python libraries installed. These tools will enable me to collect market data, build and train deep learning models, perform statistical analysis, and visualize our findings. This foundational setup is critical for any robust analytical workflow.")
    st.markdown(f"## 2. Acquiring and Preparing Multi-Asset Market Data")
    st.markdown(f"My firm, Apex Asset Management, manages a diversified multi-asset portfolio. To effectively monitor for unusual movements, I need to collect historical daily data for a broad universe of assets including equities, bonds, commodities, and currencies. I'll then transform this raw price data into daily returns and engineer additional cross-sectional features that capture market dynamics beyond individual asset movements.")
    st.markdown(f"### Market Data Acquisition and Feature Engineering")

    # UI: Date range selection for data acquisition and training split
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date for Market Data", value=datetime.date(2008, 1, 1), min_value=datetime.date(1990, 1, 1), max_value=datetime.date(2023, 12, 31))
    with col2:
        end_date = st.date_input("End Date for Market Data", value=datetime.date(2024, 12, 31), min_value=datetime.date(2008, 1, 2), max_value=datetime.date(2024, 12, 31))
    
    train_end_date = st.date_input("End Date for AE Training Period (Normal Market Behavior)", value=datetime.date(2023, 1, 1), min_value=start_date + datetime.timedelta(days=365*2), max_value=end_date - datetime.timedelta(days=30))

    # UI: Button to trigger data acquisition and preprocessing
    if st.button("Acquire & Prepare Market Data"):
        with st.spinner("Downloading and processing market data..."):
            # Invocation: acquire_and_preprocess_market_data
            market_returns, known_anomaly_dates = acquire_and_preprocess_market_data(start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
            # Invocation: prepare_data_for_ae
            X_train, X_test, scaler, train_mask, test_mask, ae_features = prepare_data_for_ae(market_returns, train_end_date=train_end_date.strftime('%Y-%m-%d'))

            # Update st.session_state
            st.session_state.market_returns = market_returns
            st.session_state.known_anomaly_dates = known_anomaly_dates
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.scaler = scaler
            st.session_state.train_mask = train_mask
            st.session_state.test_mask = test_mask
            st.session_state.ae_features = ae_features
            st.session_state.input_dim = X_train.shape[1] # Set input_dim from X_train
            st.success("Market data prepared successfully!")
    
    # Conditional display of prepared data
    if st.session_state.market_returns is not None:
        st.markdown(f"### Prepared Market Data (First 5 Rows)")
        st.dataframe(st.session_state.market_returns.head())
        st.markdown(f"Dataset prepared: {st.session_state.market_returns.shape[0]} days x {st.session_state.market_returns.shape[1]} features")
        st.markdown(f"Training set: {st.session_state.X_train.shape[0]} days, {st.session_state.X_train.shape[1]} features")
        st.markdown(f"Test set: {st.session_state.X_test.shape[0]} days")
        st.markdown(f"Features used for Autoencoder: {st.session_state.ae_features}")
        
        st.markdown(f"### Known Anomaly Dates")
        for d in st.session_state.known_anomaly_dates:
            d_str = d.strftime('%Y-%m-%d')
            if d in st.session_state.market_returns.index:
                st.markdown(f" - {d_str}: SP500 return = {st.session_state.market_returns.loc[d, 'SP500']:.2%}")
            else:
                st.markdown(f" - {d_str}: Date not found in dataset.")

# --- Page: Autoencoder Model & Training ---
elif st.session_state.current_page == "Autoencoder Model & Training":
    clear_matplotlib_cache()
    st.header("Building & Training the Autoencoder")
    st.markdown(f"## 3. Building an Autoencoder to Model 'Normal' Market Dynamics")
    st.markdown(f"As a Quant Analyst, my goal is to build a model that can discern the intricate, non-linear relationships and typical co-movements among assets during 'normal' market conditions. An Autoencoder (AE) is perfect for this. It's an unsupervised neural network that learns to compress its input into a low-dimensional 'latent space' and then reconstruct it. By forcing the network to reconstruct its own input, it learns the most salient features and patterns.")
    st.markdown(f"The architecture will be symmetric: an encoder maps the high-dimensional input to a bottleneck (latent space), and a decoder maps from the bottleneck back to the original input dimension. The critical part is the bottleneck, which acts as a 'non-linear PCA' by forcing the model to learn a compact representation of the data. If the model can reconstruct data points well, they are considered 'normal'. If it struggles, they are potentially anomalous.")
    st.markdown(f"### Autoencoder Architecture Definition")
    
    # Formula: AE mapping
    st.markdown(r"The autoencoder learns a mapping $E: \mathcal{X} \rightarrow \mathcal{Z}$ for the encoder and $D: \mathcal{Z} \rightarrow \mathcal{X}$ for the decoder, where $\mathcal{X}$ is the input space and $\mathcal{Z}$ is the latent space. The objective is to minimize the reconstruction error $\mathcal{L}_{AE}(x_i, D(E(x_i)))$.")
    st.markdown(r"where $\mathcal{X}$ is the input data space, $\mathcal{Z}$ is the latent (bottleneck) space, $E$ is the encoder function, and $D$ is the decoder function. $x_i$ is an input data point.")

    if st.session_state.input_dim is None:
        st.warning("Please acquire and prepare data first on the 'Data Acquisition & Preparation' page.")
    else:
        st.markdown(f"Input Dimension (Number of Features): **{st.session_state.input_dim}**")
        
        # UI: Hyperparameters for Autoencoder
        encoding_dim = st.slider("Bottleneck (Latent) Dimension", min_value=1, max_value=st.session_state.input_dim - 1, value=4, step=1)
        hidden_dims_options = ['32, 16', '64, 32', '16, 8'] # Simplify for Streamlit
        selected_hidden_dims_str = st.selectbox("Hidden Layer Dimensions (Encoder/Decoder)", options=hidden_dims_options, index=0)
        hidden_dims = [int(d) for d in selected_hidden_dims_str.split(', ')]
        dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
        epochs = st.number_input("Training Epochs", min_value=10, max_value=200, value=100, step=10)
        batch_size = st.number_input("Batch Size", min_value=16, max_value=256, value=64, step=16)
        patience = st.number_input("Early Stopping Patience", min_value=5, max_value=50, value=10, step=5)


        # UI: Button to trigger AE building and training
        if st.button("Build & Train Autoencoder"):
            with st.spinner("Building and training the Autoencoder model..."):
                # Invocation: build_autoencoder
                autoencoder, encoder = build_autoencoder(st.session_state.input_dim, encoding_dim=encoding_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate)
                
                # Capture model summary output
                string_io = StringIO()
                autoencoder.summary(print_fn=lambda x: string_io.write(x + '\n'))
                model_summary = string_io.getvalue()
                string_io.close()

                # Update st.session_state with model and summary
                st.session_state.autoencoder = autoencoder
                st.session_state.encoder = encoder
                st.session_state.model_summary = model_summary
                
                # Invocation: train_autoencoder
                ae_training_history = train_autoencoder(autoencoder, st.session_state.X_train, epochs=epochs, batch_size=batch_size, patience=patience)
                st.session_state.ae_training_history = ae_training_history
                st.success("Autoencoder built and trained!")
        
        # Conditional display of model summary and training results
        if st.session_state.autoencoder is not None and st.session_state.model_summary is not None:
            st.markdown(f"### Autoencoder Model Summary")
            st.code(st.session_state.model_summary)
            
            st.markdown(f"### Explanation of Autoencoder Architecture")
            st.markdown(f"The autoencoder's goal is to minimize the Mean Squared Error (MSE) between its input $x$ and its reconstructed output $\hat{x}$. This forces the network to learn an efficient, compressed representation of the input data in the latent space.")
            # Formula: AE Reconstruction Loss
            st.markdown(r"$$ \n\mathcal{L}_{AE} = \frac{1}{N} \sum_{i=1}^{N} \|x_i - \hat{x}_i\|^2\n$$")
            st.markdown(r"where $x_i$ is the original input, and $\hat{x}_i = D(E(x_i))$ is the reconstructed output. $N$ is the number of samples.")
            st.markdown(f"For a Risk Manager, this means the model learns the 'normal' manifold of market movements. Data points that lie far from this manifold will be poorly reconstructed, leading to a high reconstruction error, which serves as our anomaly score. The `encoding_dim` (bottleneck size) is a critical hyperparameter:")
            st.markdown(f"*   Too large: The autoencoder might simply memorize the input, including anomalies, leading to low reconstruction error for everything and no useful signal.")
            st.markdown(f"*   Too small: Too much information is lost, and everything might look anomalous.")
            st.markdown(f"The chosen size (e.g., **{encoding_dim}** for financial data with ~{st.session_state.input_dim} features) typically forces the model to capture only the most important structural relationships.")

            st.markdown(f"## 4. Training the Autoencoder on Normal Market Behavior")
            st.markdown(f"Now that the autoencoder architecture is defined, I will train it using the historical market return data from our designated 'normal' period. It's crucial that the model learns the patterns of typical market behavior. I'll use `EarlyStopping` to prevent overfitting and ensure the model generalizes well, stopping training when the validation loss no longer improves.")
            
            st.markdown(f"### Visualization of Training Progress")
            if st.session_state.ae_training_history is not None:
                # Invocation: plot_training_history
                plot_training_history(st.session_state.ae_training_history)
                st.pyplot(plt.gcf()) # Capture and display the last generated matplotlib figure
                plt.close('all') # Clear the figure
            else:
                st.info("Train the autoencoder to see the training progress plot.")
            
            st.markdown(f"### Explanation of Training Results")
            st.markdown(f"The training loss curves demonstrate how well the autoencoder learned to reconstruct 'normal' market behavior. A converging loss indicates that the model is successfully identifying underlying patterns. The `val_loss` is crucial: if it starts increasing while `loss` continues to decrease, it's a sign of overfitting to the training data, which `EarlyStopping` helps to mitigate by restoring the best weights. For me, the Quant Analyst, this plot confirms the model has effectively learned a baseline of 'normalcy,' a prerequisite for robust anomaly detection.")

# --- Page: Anomaly Detection & Thresholding ---
elif st.session_state.current_page == "Anomaly Detection & Thresholding":
    clear_matplotlib_cache()
    st.header("Detecting Anomalies: Identifying Unusual Market Movements with Reconstruction Error")
    st.markdown(f"## 5. Detecting Anomalies: Identifying Unusual Market Movements with Reconstruction Error")
    st.markdown(f"With a trained autoencoder, I can now move to the core task: detecting unusual market movements. This involves using the autoencoder to reconstruct both training and test data and then calculating the 'reconstruction error' for each day. Days with significantly higher reconstruction errors are flagged as potential anomalies. I will set a statistical threshold based on the distribution of errors from the training set, which represents normal behavior.")
    st.markdown(f"### Calculating Reconstruction Errors and Setting Anomaly Thresholds")
    st.markdown(f"The reconstruction error $e_i$ for a given input $x_i$ is calculated as the Mean Squared Error (MSE) between the original input and its reconstruction $\hat{x}_i$: ")
    # Formula: Reconstruction Error
    st.markdown(r"$$ e_i = \frac{1}{F} \sum_{j=1}^{F} (x_{ij} - \hat{x}_{ij})^2 $$")
    st.markdown(r"where $F$ is the number of features, $x_{ij}$ is the $j$-th feature of the $i$-th observation, and $\hat{x}_{ij}$ is its reconstructed value.")
    st.markdown(f"To classify an observation as anomalous, its reconstruction error $e_i$ must exceed a predefined threshold $\tau$.")
    # Formula: Anomaly Classification
    st.markdown(r"$$ \text{Anomaly}_i = \begin{cases} 1 & \text{if } e_i > \tau \\ 0 & \text{otherwise} \end{cases} $$")
    st.markdown(r"where $e_i$ is the reconstruction error for observation $i$, and $\tau$ is the anomaly threshold.")
    st.markdown(f"For thresholding, I will use percentile-based methods on the training errors, as they are simple and robust. I will consider the 95th and 99th percentiles to represent moderate and extreme anomaly thresholds, respectively.")

    if st.session_state.autoencoder is None or st.session_state.X_train is None:
        st.warning("Please build and train the Autoencoder first on the 'Autoencoder Model & Training' page.")
    else:
        # UI: Select percentiles for anomaly thresholds
        percentiles_selected = st.multiselect("Select Anomaly Threshold Percentiles", options=[90, 95, 99, 99.5], default=[95, 99])
        
        # UI: Button to calculate errors and thresholds
        if st.button("Calculate Reconstruction Errors & Set Thresholds"):
            if not percentiles_selected:
                st.error("Please select at least one percentile for anomaly thresholds.")
            else:
                with st.spinner("Calculating errors and thresholds..."):
                    # Invocation: calculate_reconstruction_errors
                    train_recon_errors = calculate_reconstruction_errors(st.session_state.autoencoder, st.session_state.X_train)
                    test_recon_errors = calculate_reconstruction_errors(st.session_state.autoencoder, st.session_state.X_test)
                    # Invocation: set_anomaly_thresholds
                    anomaly_thresholds = set_anomaly_thresholds(train_recon_errors, percentiles=percentiles_selected)

                    # Update st.session_state
                    st.session_state.train_recon_errors = train_recon_errors
                    st.session_state.test_recon_errors = test_recon_errors
                    st.session_state.anomaly_thresholds = anomaly_thresholds

                    # Flag anomalies and create results df
                    test_dates = st.session_state.market_returns.index[st.session_state.test_mask]
                    
                    # Ensure flags are created even if a percentile is not selected (default to False)
                    anomaly_flags_99 = test_recon_errors > anomaly_thresholds.get(99, np.inf)
                    anomaly_flags_95 = test_recon_errors > anomaly_thresholds.get(95, np.inf)

                    anomaly_results_df = pd.DataFrame({
                        'date': test_dates,
                        'recon_error': test_recon_errors,
                        'SP500_ret': st.session_state.market_returns.loc[st.session_state.test_mask, 'SP500'].values,
                        'VIX': st.session_state.market_returns.loc[st.session_state.test_mask, 'VIX'].values,
                        'anomaly_flag_99': anomaly_flags_99,
                        'anomaly_flag_95': anomaly_flags_95
                    }).set_index('date')
                    st.session_state.anomaly_results_df = anomaly_results_df
                    st.session_state.top_anomalies_df = anomaly_results_df.nlargest(10, 'recon_error')
                    st.success("Anomalies calculated and flagged!")
        
        # Conditional display of error stats, distribution, time series, and top anomalies
        if st.session_state.anomaly_results_df is not None:
            st.markdown(f"### Reconstruction Error Statistics")
            st.markdown(f"Training error stats: Mean={np.mean(st.session_state.train_recon_errors):.4f}, Std={np.std(st.session_state.train_recon_errors):.4f}")
            for p, t in st.session_state.anomaly_thresholds.items():
                st.markdown(f"Threshold ({p}th percentile): {t:.4f}")
            
            st.markdown(f"### Visualization of Reconstruction Error Distribution")
            # Invocation: plot_error_distribution
            plot_error_distribution(st.session_state.train_recon_errors, st.session_state.anomaly_thresholds)
            st.pyplot(plt.gcf()) # Capture and display the last generated matplotlib figure
            plt.close('all') # Clear the figure

            st.markdown(f"### Explanation of Anomaly Detection")
            st.markdown(f"This step provides the quantitative basis for anomaly detection. The histogram of training errors typically shows a heavy-tailed distribution, where the bulk of 'normal' market days have low errors, and extreme events correspond to the long tail. By setting thresholds at the {percentiles_selected[0]}th or {percentiles_selected[1]}th percentile (example values), I am defining what constitutes a statistically significant deviation from normal. For Apex Asset Management, this allows for a consistent and data-driven approach to flagging market events that warrant further investigation, moving beyond arbitrary fixed thresholds.")

            st.markdown(f"### Reconstruction Error Time Series")
            # Invocation: plot_anomaly_time_series
            plot_anomaly_time_series(st.session_state.anomaly_results_df, st.session_state.anomaly_thresholds, st.session_state.known_anomaly_dates)
            st.pyplot(plt.gcf()) # Capture and display the last generated matplotlib figure
            plt.close('all') # Clear the figure

            st.markdown(f"### Top 10 Anomalous Days by Reconstruction Error")
            st.dataframe(st.session_state.top_anomalies_df[['recon_error', 'SP500_ret', 'VIX', 'anomaly_flag_99']])

# --- Page: Anomaly Insights & Latent Space ---
elif st.session_state.current_page == "Anomaly Insights & Latent Space":
    clear_matplotlib_cache()
    st.header("Deep Dive into Anomalies & Latent Structure")
    st.markdown(f"## 6. Visualizing Anomalies and Their Drivers for Deeper Insight")
    st.markdown(f"Detecting an anomaly is only the first step. As a Quant Analyst, I need to visualize these flagged dates in the context of broader market movements and, crucially, understand *what* specifically made them anomalous. This helps in root cause analysis and informs risk mitigation strategies. I'll create a dual-panel time series plot and then drill down into per-feature errors for the most anomalous days.")
    st.markdown(f"### Per-Feature Error Analysis (Anomaly Gallery & Heatmap)")
    st.markdown(f"A high total reconstruction error tells me *that* something is anomalous, but not *what*. Per-feature error analysis helps pinpoint the specific assets or features that contributed most to the anomaly. The per-feature reconstruction error for observation $i$ and feature $j$ is simply $(x_{ij} - \hat{x}_{ij})^2$. Summing this over selected anomalous days can highlight consistently problematic features.")
    # Formula: Per-Feature Error
    st.markdown(r"$$ \text{Per-Feature Error}_{ij} = (x_{ij} - \hat{x}_{ij})^2 $$")
    st.markdown(r"where $x_{ij}$ is the original value of feature $j$ for observation $i$, and $\hat{x}_{ij}$ is its reconstructed value.")

    if st.session_state.anomaly_results_df is None or st.session_state.autoencoder is None:
        st.warning("Please detect anomalies first on the 'Anomaly Detection & Thresholding' page.")
    else:
        # UI: Select anomalous dates for detailed analysis
        anomaly_dates_for_analysis = st.multiselect(
            "Select specific anomalous dates for per-feature analysis:",
            options=st.session_state.top_anomalies_df.index.strftime('%Y-%m-%d').tolist(), # Display as string
            default=st.session_state.top_anomalies_df.index[:min(5, len(st.session_state.top_anomalies_df.index))].strftime('%Y-%m-%d').tolist()
        )
        
        # UI: Button to analyze per-feature errors
        if st.button("Analyze Per-Feature Errors"):
            if anomaly_dates_for_analysis:
                with st.spinner("Analyzing per-feature errors..."):
                    # Invocation: analyze_per_feature_errors
                    analyze_per_feature_errors(
                        st.session_state.autoencoder, 
                        st.session_state.X_test, 
                        st.session_state.market_returns, 
                        pd.to_datetime(anomaly_dates_for_analysis), # Ensure dates are datetime objects for the function
                        st.session_state.scaler, 
                        st.session_state.ae_features
                    )
                    st.markdown("### Anomaly Gallery: Original vs. Reconstructed Features (Scaled)")
                    st.pyplot(plt.gcf()) # Capture and display the last generated figure (Anomaly Gallery)
                    plt.close('all') # Clear the figure
                    st.markdown("### Per-Feature Reconstruction Error Heatmap")
                    st.pyplot(plt.gcf()) # Capture and display the next generated figure (Heatmap)
                    plt.close('all') # Clear the figure
            else:
                st.info("Please select at least one anomalous date to analyze.")
        
        st.markdown(f"### Explanation of Anomaly Visualizations and Drivers")
        st.markdown(f"The dual-panel time series plot (from the previous page) gives me, the Quant Analyst, an immediate visual context. I can see the market's reaction (S&P 500 returns) alongside the autoencoder's reconstruction error. Crucially, the flagged red dots indicate days where our autoencoder detected significant anomalies. This allows for qualitative validation against known market events or periods of high volatility.")
        st.markdown(f"The 'anomaly gallery' (original vs. reconstructed feature plot) and the heatmap are invaluable for triage. They answer the critical question: *what* was anomalous? For example, if on a particular day, bond returns were severely poorly reconstructed while equity returns were fine, it might signal an unusual bond-equity correlation breakdown, rather than a broad market crash. This level of detail aids in root cause analysis and proactive risk mitigation.")
        st.markdown(f"**Practitioner Warning:**")
        st.markdown(r"Anomaly $\neq$ Crisis.") # Split due to strict formula rule
        st.markdown(f"It's essential for me to remember that not every flagged day is a market crisis. Some might be 'structural anomalies' (e.g., cross-asset correlations shifting subtly), data quality issues, or model artifacts. The autoencoder detects *statistical* anomalies; human judgment is non-negotiable to determine financial meaningfulness and decide on appropriate actions. This human-in-the-loop interpretation is vital for production deployment at Apex Asset Management.")

        st.markdown(f"## 7. Uncovering Hidden Market Regimes: Latent Space Analysis")
        st.markdown(f"Beyond just flagging anomalies, I want to understand what patterns the autoencoder has learned about 'normal' market structure. Visualizing the compressed 'latent space' can reveal how different market regimes or conditions cluster. By training a specialized autoencoder with a 2D latent space and coloring the points by a market regime proxy (like VIX Daily Change), I can gain insights into the model's structural learning.")
        st.markdown(f"### Training a 2D Latent Space Autoencoder and Visualization")
        
        # UI: Button to visualize latent space
        if st.button("Visualize Latent Space (2D)"):
            with st.spinner("Building and visualizing 2D latent space..."):
                # Invocation: build_and_visualize_latent_space
                build_and_visualize_latent_space(
                    st.session_state.X_train, 
                    st.session_state.X_test, 
                    st.session_state.market_returns, 
                    st.session_state.train_mask, 
                    st.session_state.test_mask, 
                    st.session_state.ae_features
                )
                st.pyplot(plt.gcf()) # Capture and display the last generated matplotlib figure
                plt.close('all') # Clear the figure
                st.markdown("*(Plot generated above shows the 2D latent space colored by VIX Daily Change, with anomalies highlighted.)*")

        st.markdown(f"### Explanation of Latent Space Visualization")
        st.markdown(f"This visualization is a powerful way for me to assess if the autoencoder has learned meaningful, structural relationships in the market data. By projecting the high-dimensional market returns into a 2D latent space and coloring points by a proxy for market regime (like VIX Daily Change), I can observe patterns. For instance, if periods of high VIX change (indicating market stress) cluster together, it suggests the autoencoder is capturing these underlying structures. This provides confidence that the model isn't just memorizing data but is truly learning the 'geometry' of normal market behavior, allowing it to better identify deviations.")

# --- Page: Model Benchmarking & VAE Extension ---
elif st.session_state.current_page == "Model Benchmarking & VAE Extension":
    clear_matplotlib_cache()
    st.header("Benchmarking Anomaly Detection & VAE for Generative Finance")
    st.markdown(f"## 8. Benchmarking Anomaly Detection: Autoencoder vs. Baselines")
    st.markdown(f"To fully justify the autoencoder's deployment, I need to compare its performance against simpler, more traditional anomaly detection methods. This benchmarking provides critical context and helps quantify the added value of a deep learning approach for Apex Asset Management. I'll compare it with Isolation Forest and Mahalanobis distance.")
    st.markdown(f"For evaluation, I will use the `known_anomaly_dates` as a sparse 'ground truth' to calculate ROC AUC and Average Precision scores. It's important to note that market data isn't perfectly labeled, so these metrics provide a directional comparison rather than absolute performance.")
    st.markdown(f"### Implementing and Comparing Baseline Models")
    st.markdown(f"**Isolation Forest:** This algorithm isolates anomalies by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. Anomalies are points that require fewer splits to be isolated. The anomaly score $s(x)$ for a point $x$ is:")
    # Formula: Isolation Forest Anomaly Score
    st.markdown(r"$$ s(x) = 2^{-\frac{E[h(x)]}{c(n)}} $$")
    st.markdown(r"where $E[h(x)]$ is the average path length to isolate $x$, and $c(n)$ is the expected path length for a dataset of size $n$.")
    st.markdown(f"**Mahalanobis Distance:** This is a measure of the distance between a point and a distribution. It is a multi-dimensional generalization of the idea of measuring how many standard deviations away a point is from the mean of a distribution. For a point $x$ and a distribution with mean $\mu$ and covariance matrix $\Sigma$, the Mahalanobis distance $D_M(x)$ is:")
    # Formula: Mahalanobis Distance
    st.markdown(r"$$ D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)} $$")
    st.markdown(r"where $x$ is the data point, $\mu$ is the mean of the distribution, and $\Sigma^{-1}$ is the inverse covariance matrix. Higher values indicate greater deviation from the normal distribution.")

    if st.session_state.autoencoder is None or st.session_state.anomaly_results_df is None:
        st.warning("Please ensure Autoencoder is trained and anomalies are detected to compare baselines.")
    else:
        # UI: Button to compare anomaly detectors
        if st.button("Compare Anomaly Detectors"):
            with st.spinner("Running baseline comparisons..."):
                # Invocation: compare_anomaly_detectors
                compare_anomaly_detectors(
                    st.session_state.X_train, 
                    st.session_state.X_test, 
                    st.session_state.anomaly_results_df, 
                    st.session_state.known_anomaly_dates, 
                    st.session_state.autoencoder, 
                    st.session_state.test_recon_errors
                )
                st.pyplot(plt.gcf()) # Capture and display the last generated matplotlib figure (Precision-Recall)
                plt.close('all') # Clear the figure
                st.markdown("*(Precision-Recall curves displayed above if known anomalies are present in the test set.)*")
        
        st.markdown(f"### Explanation of Baseline Comparison")
        st.markdown(f"The comparison against Isolation Forest and Mahalanobis distance is vital for a Quant Analyst like me. While the autoencoder might capture more complex, non-linear dependencies, it's good practice to understand how it stacks up against simpler, faster methods. Isolation Forest is a tree-based method, robust to high-dimensional data, while Mahalanobis distance is a statistical measure assuming multivariate normality. The Precision-Recall curves, if calculable, provide a visual summary of the trade-off between identifying all true anomalies (recall) and minimizing false positives (precision) across various thresholds. This exercise helps Apex Asset Management choose the most appropriate tool for a given anomaly detection task, balancing complexity, interpretability, and performance.")

    st.markdown(f"---")
    st.markdown(f"## 9. Beyond Detection: Introduction to Variational Autoencoders for Generative Finance")
    st.markdown(f"While our autoencoder is excellent for anomaly detection, there's a powerful extension called the Variational Autoencoder (VAE) that allows for generative modeling. For an investment professional, this capability opens doors to generating synthetic market data for stress testing, scenario simulation, and even data augmentation. It's a conceptual bridge from detecting anomalies to creating realistic data.")
    st.markdown(f"### Variational Autoencoder (VAE) Architecture and Loss")
    st.markdown(f"A VAE differs from a standard autoencoder by mapping the input to a probability distribution in the latent space (mean and variance) rather than a fixed point. This allows for smooth and continuous latent spaces. The VAE's loss function combines the reconstruction error with a Kullback-Leibler (KL) divergence term, which acts as a regularizer, forcing the latent distribution to be close to a standard normal distribution.")
    st.markdown(f"The VAE's loss function $\mathcal{L}_{VAE}$ is composed of two parts:")
    st.markdown(f"1. **Reconstruction Loss:** Measures how well the input is reconstructed, typically MSE.")
    st.markdown(r"$\mathcal{L}_{recon} = \|x_i - D(\tilde{z}_i)\|^2$ where $\tilde{z}_i$ is a sample from the latent distribution.")
    st.markdown(f"2. **KL Divergence Loss:** A regularization term that measures the difference between the learned latent distribution $q(z|x_i) = \mathcal{N}(\mu_i, \Sigma_i)$ and a prior standard normal distribution $p(z) = \mathcal{N}(0, I)$.")
    # Formula: KL Divergence Loss
    st.markdown(r"$$ \mathcal{L}_{KL} = -\frac{1}{2} \sum_{j=1}^{d} (1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2) $$")
    st.markdown(r"where $d$ is the latent dimension, and $\mu_j$ and $\sigma_j^2$ are the mean and variance of the $j$-th latent dimension.")
    st.markdown(f"The total VAE loss is:")
    # Formula: Total VAE Loss
    st.markdown(r"$$ \mathcal{L}_{VAE} = \mathcal{L}_{recon} + \beta \mathcal{L}_{KL} $$")
    st.markdown(r"where $\mathcal{L}_{recon}$ is the reconstruction loss, $\mathcal{L}_{KL}$ is the KL divergence loss, and $\beta$ is a weighting factor.")

    st.markdown(f"### VAE Implementation (Brief Demonstration)")
    if st.session_state.input_dim is None or st.session_state.X_train is None:
        st.warning("Please ensure data is prepared to demonstrate VAE.")
    else:
        # UI: Button to build VAE model
        if st.button("Build VAE Model"):
            with st.spinner("Building VAE model..."):
                # Invocation: build_vae
                vae_model, vae_encoder, vae_decoder = build_vae(st.session_state.input_dim, latent_dim=4)
                st.session_state.vae_model = vae_model
                st.session_state.vae_encoder = vae_encoder
                st.session_state.vae_decoder = vae_decoder
                
                # Capture model summary output
                string_io_vae = StringIO()
                vae_model.summary(print_fn=lambda x: string_io_vae.write(x + '\n'))
                vae_model_summary = string_io_vae.getvalue()
                string_io_vae.close()
                st.session_state.vae_model_summary = vae_model_summary
                st.session_state.vae_trained_flag = False # Reset trained flag on build
                st.success("VAE model built!")
        
        # Conditional display of VAE model summary and training button
        if st.session_state.vae_model is not None and st.session_state.vae_model_summary is not None:
            st.markdown("### VAE Model Summary:")
            st.code(st.session_state.vae_model_summary)
            
            if st.button("Train VAE (Demonstration)"):
                with st.spinner("Training VAE..."):
                    # Invocation: vae_model.fit (direct method call as in source.py)
                    st.session_state.vae_model.fit(
                        st.session_state.X_train, st.session_state.X_train, 
                        epochs=50, batch_size=64, validation_split=0.15, verbose=0,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
                    )
                    st.session_state.vae_trained_flag = True
                    st.success("VAE trained successfully.")
            elif st.session_state.vae_trained_flag:
                st.info("VAE already trained.")
            else:
                st.info("Click 'Train VAE (Demonstration)' to start training.")

        st.markdown(f"### Explanation of VAE for Finance")
        st.markdown(f"For Apex Asset Management, the VAE is conceptually significant:")
        st.markdown(f"*   **Generative Capabilities:** Unlike the standard autoencoder which is purely discriminative (for anomaly detection), the VAE can *generate* new, plausible data points by sampling from its smooth latent space and passing them through the decoder. This is crucial for creating synthetic financial data.")
        st.markdown(f"*   **Synthetic Data Generation:** This capability can be used for:")
        st.markdown(f"    *   **Stress Testing:** Generating extreme but realistic market scenarios for portfolio stress testing.")
        st.markdown(f"    *   **Scenario Simulation:** Creating various market scenarios to evaluate strategy performance.")
        st.markdown(f"    *   **Data Augmentation:** Expanding limited datasets, particularly for rare events, to improve model training.")
        st.markdown(f"This brief demonstration of the VAE highlights how deep learning architectures can extend from anomaly detection to advanced generative applications, providing even more sophisticated tools for financial professionals.")
```
