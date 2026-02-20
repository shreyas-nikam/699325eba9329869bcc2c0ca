# app.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt  # For capturing plots
import seaborn as sns  # Used by source.py plots (import here for consistency)
import tensorflow as tf  # For EarlyStopping callback
from io import StringIO  # For capturing model summary output
import sys
from contextlib import redirect_stdout

# Import all functions from source.py
from source import *

# Set matplotlib to non-interactive mode for Streamlit
plt.ioff()

# Monkeypatch plt.show() to do nothing (Streamlit handles figure display)
plt.show = lambda: None

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Market Surveillance (Autoencoder)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Session State Initialization
# ----------------------------

# Navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = "Build the Market Lens"

# Data acquisition / prep
if "market_returns" not in st.session_state:
    st.session_state.market_returns = None
if "known_anomaly_dates" not in st.session_state:
    st.session_state.known_anomaly_dates = None
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "train_mask" not in st.session_state:
    st.session_state.train_mask = None
if "test_mask" not in st.session_state:
    st.session_state.test_mask = None
if "ae_features" not in st.session_state:
    st.session_state.ae_features = None
if "input_dim" not in st.session_state:
    st.session_state.input_dim = None

# Autoencoder model & training
if "autoencoder" not in st.session_state:
    st.session_state.autoencoder = None
if "encoder" not in st.session_state:
    st.session_state.encoder = None
if "model_summary" not in st.session_state:
    st.session_state.model_summary = None
if "ae_training_history" not in st.session_state:
    st.session_state.ae_training_history = None

# Anomaly detection
if "train_recon_errors" not in st.session_state:
    st.session_state.train_recon_errors = None
if "test_recon_errors" not in st.session_state:
    st.session_state.test_recon_errors = None
if "anomaly_thresholds" not in st.session_state:
    st.session_state.anomaly_thresholds = None
if "anomaly_results_df" not in st.session_state:
    st.session_state.anomaly_results_df = None
if "top_anomalies_df" not in st.session_state:
    st.session_state.top_anomalies_df = None

# VAE extension
if "vae_model" not in st.session_state:
    st.session_state.vae_model = None
if "vae_encoder" not in st.session_state:
    st.session_state.vae_encoder = None
if "vae_decoder" not in st.session_state:
    st.session_state.vae_decoder = None
if "vae_model_summary" not in st.session_state:
    st.session_state.vae_model_summary = None
if "vae_trained_flag" not in st.session_state:
    st.session_state.vae_trained_flag = False


# ----------------------------
# Helpers
# ----------------------------
def clear_matplotlib_cache():
    """Clear matplotlib figures to avoid stale plots across reruns/pages."""
    plt.close("all")


def display_and_clear_all_figures():
    """
    Render ALL matplotlib figures created (e.g., by source.py),
    then close them to avoid caching artifacts.
    """
    fignums = plt.get_fignums()
    if fignums:
        for num in fignums:
            fig = plt.figure(num)
            st.pyplot(fig)
        plt.close("all")
        return len(fignums)
    else:
        return 0


def assumptions_box(lines):
    st.markdown("#### Assumptions & Limits (read before interpreting results)")
    st.markdown("\n".join([f"- {x}" for x in lines]))


def provenance_box(lines):
    st.markdown("#### How these numbers are computed (provenance)")
    st.markdown("\n".join([f"- {x}" for x in lines]))


def severity_legend():
    st.markdown("#### Alert tiers (policy semantics)")
    st.markdown(
        "- **Watchlist (95th percentile)**: flag and triage; likely higher false positives.\n"
        "- **Escalate (99th percentile)**: investigate promptly; lower frequency, higher severity.\n"
        "- **Critical (99.5th percentile)**: rare-event style alert; treat as potential regime break."
    )


st.title("QuLab: Lab 15 - Autoencoders for Anomaly Detection")
st.divider()
# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.sidebar.title("Navigation")

page_selection = st.sidebar.selectbox(
    "Go to",
    [
        "Build the Market Lens",
        "Learn Normal Market Structure",
        "Turn Deviation into Alerts",
        "Explain the Alert & Map Regimes",
        "Validate the Alarm & Scenario Thinking",
    ],
)
st.session_state.current_page = page_selection

st.sidebar.markdown("---")
st.sidebar.markdown("### Decision hygiene")
st.sidebar.markdown(
    "- Anomaly ≠ crisis\n"
    "- Anomaly ≠ forecast\n"
    "- Always verify data definitions and regimes"
)

# ============================================================
# Page 1: Data Acquisition & Preparation
# ============================================================
if st.session_state.current_page == "Build the Market Lens":
    clear_matplotlib_cache()

    st.title("Build the Market Lens: What Data Defines ‘Normal’?")

    st.markdown("## 1. Setting the Stage: What this app will help you do")
    st.markdown(
        "This application is designed for investment professionals to **monitor for unusual market behavior** using a "
        "transparent, auditable approach:\n\n"
        "1) Define a **market lens** (assets + engineered features)\n"
        "2) Learn a baseline view of **normal cross-asset structure**\n"
        "3) Flag days that deviate materially from that baseline\n"
        "4) Explain *what drove* each alert (drivers) and *what regime* it resembles (regime map)\n\n"
        "This is **surveillance and triage**, not a return forecasting tool."
    )

    st.info(
        "Learning checkpoint: The single most important governance choice is what you call 'normal'—"
        "because the model will treat whatever it sees in that window as baseline behavior."
    )

    st.markdown(
        "## 2. Data window and training period")
    st.markdown(
        "Apex Asset Management manages a diversified **multi-asset portfolio**. To monitor for unusual movements, "
        "we use a feature set built from historical daily data across assets and cross-sectional signals.\n\n"
        "The analysis uses:\n"
        "- A **history window** from 2008-01-01 to 2024-12-31 (covers multiple market regimes)\n"
        "- A **training cutoff** at 2023-01-01 that defines what period the model learns as *normal*"
    )

    assumptions_box(
        [
            "Daily frequency can miss intra-day microstructure events; interpret alerts as daily regime breaks.",
            "Feature definitions and data cleaning rules materially affect what gets flagged—review the feature dictionary.",
            "If the training window includes crisis regimes, the system may treat crisis dynamics as ‘normal’ and become less sensitive to stress."
        ]
    )

    st.markdown("### Market data window")
    # Fixed date range - matching default values from temp.py
    start_date = datetime.date(2008, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    train_end_date = datetime.date(2023, 1, 1)

    st.info(
        f"**Data Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n\n"
        f"**Training Cutoff (normal learning period):** {train_end_date.strftime('%Y-%m-%d')}\n\n"
        f"The model will learn 'normal' market behavior from data up to {train_end_date.strftime('%Y-%m-%d')} "
        f"and flag anomalies in the period after that date."
    )

    with st.expander("What’s in the dataset (feature dictionary & transformations)", expanded=False):
        st.markdown(
            "**You should be able to answer these before trusting alerts:**\n\n"
            "- What asset universe is included?\n"
            "- Are returns simple or log returns?\n"
            "- Which engineered features exist (dispersion, correlations, risk proxies)?\n"
            "- How are missing values handled?\n"
            "- What scaling is applied before modeling?\n\n"
            "If you have a formal feature dictionary, include it here for governance."
        )

    # Checkpoint question
    st.markdown("### Checkpoint question (for intuition)")
    ans = st.radio(
        "If you include a crisis period in the ‘normal’ training window, what happens to sensitivity to future crisis-like moves?",
        ["More sensitive (flags more crisis days)",
         "Less sensitive (treats crisis dynamics as normal)", "No effect"],
        index=1,
        help="This is about what the system learns as baseline behavior.",
    )
    if ans == "Less sensitive (treats crisis dynamics as normal)":
        st.success(
            "Correct. If crisis dynamics are in the baseline, similar dynamics will no longer look unusual.")
    elif ans:
        st.info("Not quite. Including crisis behavior in baseline tends to make the system less sensitive to crisis-like patterns later.")

    if st.button("Acquire & Prepare Market Data"):
        with st.spinner("Downloading and processing market data..."):
            market_returns, known_anomaly_dates = acquire_and_preprocess_market_data(
            )

            X_train, X_test, scaler, train_mask, test_mask, ae_features = prepare_data_for_ae(
                market_returns, train_end_date=train_end_date.strftime(
                    "%Y-%m-%d")
            )

            st.session_state.market_returns = market_returns
            st.session_state.known_anomaly_dates = known_anomaly_dates
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.scaler = scaler
            st.session_state.train_mask = train_mask
            st.session_state.test_mask = test_mask
            st.session_state.ae_features = ae_features
            st.session_state.input_dim = X_train.shape[1]

        st.success("Market data prepared successfully!")

    if st.session_state.market_returns is not None:
        provenance_box(
            [
                "Market dataset is transformed into returns and engineered features (see feature dictionary).",
                "The **training set** is all dates up to the training cutoff; the **test set** is after.",
                "Features are scaled before modeling so no single feature dominates purely due to magnitude."
            ]
        )

        st.markdown("### Prepared Market Data (preview)")
        st.dataframe(st.session_state.market_returns.head())

        st.markdown(
            f"**Dataset:** {st.session_state.market_returns.shape[0]} days × "
            f"{st.session_state.market_returns.shape[1]} features"
        )
        st.markdown(
            f"**Training set (normal learning window):** {st.session_state.X_train.shape[0]} days × "
            f"{st.session_state.X_train.shape[1]} features"
        )
        st.markdown(
            f"**Test set (surveillance window):** {st.session_state.X_test.shape[0]} days")
        st.markdown(
            f"**Features used for surveillance lens:** {st.session_state.ae_features}")

        st.markdown(
            "### Reference dates for sanity checks (not exhaustive labels)")
        st.markdown(
            "These dates are used as **reference events** to sanity check whether the system flags well-known stress episodes. "
            "They are *not* ground-truth statistical anomalies."
        )
        if st.session_state.known_anomaly_dates:
            for d in st.session_state.known_anomaly_dates:
                d_str = pd.to_datetime(d).strftime("%Y-%m-%d")
                if d in st.session_state.market_returns.index:
                    st.markdown(
                        f"- **{d_str}**: SP500 return = "
                        f"{st.session_state.market_returns.loc[d, 'SP500']:.2%}"
                    )
                else:
                    st.markdown(f"- **{d_str}**: Date not found in dataset.")
        else:
            st.info("No reference dates returned by the data acquisition function.")

        st.info(
            "Micro-example: If you train through 2020–2022, elevated volatility may be learned as normal; "
            "a 2024 vol spike may no longer be flagged as unusual."
        )


# ============================================================
# Page 2: Autoencoder Model & Training
# ============================================================
elif st.session_state.current_page == "Learn Normal Market Structure":
    clear_matplotlib_cache()

    st.title("Learn ‘Normal’ Market Structure (Think: Nonlinear Factor Lens)")

    st.markdown("## 3. What the model learns (in plain finance terms)")
    st.markdown(
        "The system learns a compact representation of typical **cross-asset co-movement**. "
        "You can think of this as a flexible, nonlinear cousin of a factor model:\n\n"
        "- The **encoder** compresses today’s market snapshot into a small set of latent drivers\n"
        "- The **decoder** reconstructs the original snapshot from those drivers\n\n"
        "If the reconstruction is poor, the day is ‘unusual’ relative to learned normal structure."
    )

    st.markdown("### Autoencoder Architecture Definition")
    st.markdown(r"The encoder and decoder can be represented as:")
    st.markdown(r"- $E: \mathcal{X} \rightarrow \mathcal{Z}$")
    st.markdown(r"- $D: \mathcal{Z} \rightarrow \mathcal{X}$")
    st.markdown(
        r"The objective is to minimize reconstruction error: $\mathcal{L}_{AE}(x_i, D(E(x_i)))$.")

    st.markdown("### What is minimized (loss)")
    st.markdown(
        "The autoencoder is trained to minimize the **Mean Squared Error (MSE)** between the input $x$ and "
        "its reconstruction $\hat{x}$. This anchors what ‘fit’ means."
    )
    st.markdown(
        r"""
$$
\mathcal{L}_{AE} = \frac{1}{N}\sum_{i=1}^{N}\|x_i - \hat{x}_i\|^2
$$""")

    assumptions_box(
        [
            "This is not a forecasting objective; it learns reconstruction of today from today.",
            "A model that is too flexible may treat noise as normal (fewer alerts, potential misses).",
            "A model that is too rigid may flag routine rotations as anomalies (more alerts, higher triage burden).",
        ]
    )

    if st.session_state.input_dim is None:
        st.warning("Please prepare data first on ‘Build the Market Lens’.")
    else:
        st.markdown(
            f"**Number of input features in your lens:** {st.session_state.input_dim}")

        st.markdown(
            "## 4. Choose a baseline complexity (controls the sensitivity tradeoff)")
        encoding_dim = st.slider(
            "Number of latent drivers (think: factors the model uses)",
            min_value=1,
            max_value=max(2, st.session_state.input_dim - 1),
            value=min(4, max(2, st.session_state.input_dim - 1)),
            step=1,
            help="Smaller = simpler baseline, may flag more ‘unusual’. Larger = more detail, may treat more behavior as normal.",
        )

        with st.expander("Advanced controls (only if you need them)", expanded=False):
            hidden_dims_options = ["32, 16", "64, 32", "16, 8"]
            selected_hidden_dims_str = st.selectbox(
                "Model flexibility (how complex ‘normal structure’ can be)",
                options=hidden_dims_options,
                index=0,
                help="More flexibility can reduce false alarms but risks learning noise as normal.",
            )
            hidden_dims = [int(d.strip())
                           for d in selected_hidden_dims_str.split(",")]

            dropout_rate = st.slider(
                "Regularization strength (reduces overfitting to idiosyncrasies)",
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Higher values can stabilize learning but may underfit; lower values can overfit.",
            )

            epochs = st.number_input(
                "Max training iterations (training stops early if no improvement)",
                min_value=10,
                max_value=200,
                value=100,
                step=10,
                help="Early stopping usually prevents wasting iterations once improvement stalls.",
            )
            batch_size = st.number_input(
                "Batch size (how many days per learning step)",
                min_value=16,
                max_value=256,
                value=64,
                step=16,
                help="Typically does not change the economic meaning; it affects training stability.",
            )
            patience = st.number_input(
                "Early stopping patience (how long to wait for improvement)",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                help="Higher patience can improve fit but increases overfitting risk.",
            )

        if "hidden_dims" not in locals():
            hidden_dims = [32, 16]
        if "dropout_rate" not in locals():
            dropout_rate = 0.2
        if "epochs" not in locals():
            epochs = 100
        if "batch_size" not in locals():
            batch_size = 64
        if "patience" not in locals():
            patience = 10

        st.markdown("### Why this matters (decision relevance)")
        st.markdown(
            "- **Too flexible**: fewer alerts; may miss early regime breaks.\n"
            "- **Too rigid**: many alerts; triage burden rises and confidence can drop.\n"
            "Choose parameters consistent with how your desk escalates surveillance signals."
        )

        if st.button("Build & Train ‘Normal Structure’ Model"):
            with st.spinner("Building and training the model..."):
                autoencoder, encoder = build_autoencoder(
                    st.session_state.input_dim,
                    encoding_dim=encoding_dim,
                    hidden_dims=hidden_dims,
                    dropout_rate=dropout_rate,
                )

                string_io = StringIO()
                autoencoder.summary(
                    print_fn=lambda x: string_io.write(x + "\n"))
                st.session_state.model_summary = string_io.getvalue()
                string_io.close()

                st.session_state.autoencoder = autoencoder
                st.session_state.encoder = encoder

                st.session_state.ae_training_history = train_autoencoder(
                    autoencoder,
                    st.session_state.X_train,
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    patience=int(patience),
                )

            st.success("Model trained. Next: convert deviation into alerts.")

        if st.session_state.autoencoder is not None and st.session_state.model_summary is not None:
            provenance_box(
                [
                    "The model is trained on the training window defined on the first screen (the ‘normal’ period).",
                    "The objective is the MSE reconstruction loss shown above.",
                    "Training uses early stopping to reduce overfitting to idiosyncratic days."
                ]
            )

            st.markdown("### Transparency: Model summary (for auditability)")
            st.code(st.session_state.model_summary)

            st.markdown("### Training diagnostics (what ‘good’ looks like)")
            st.markdown(
                "- **Acceptable**: training and validation loss converge and stabilize.\n"
                "- **Watch-out**: validation loss rises while training loss falls → the system may become unreliable out of sample."
            )

            if st.session_state.ae_training_history is not None:
                plot_training_history(st.session_state.ae_training_history)
                display_and_clear_all_figures()

            st.info(
                "Micro-example: A 4-driver lens may capture equity, rates, credit, and volatility regimes; "
                "too many drivers can start fitting idiosyncratic product noise."
            )


# ============================================================
# Page 3: Anomaly Detection & Thresholding
# ============================================================
elif st.session_state.current_page == "Turn Deviation into Alerts":
    clear_matplotlib_cache()

    st.title("Turn Deviation into Alerts: Anomaly Score → Threshold Policy")

    st.markdown("## 5. From reconstruction error to an anomaly score")
    st.markdown(
        "With a trained baseline model, we compute **reconstruction error** each day. "
        "Large errors indicate the day’s cross-asset pattern did not resemble the learned normal structure.\n\n"
        "Next, we translate the error distribution into an **alert policy** using percentile thresholds."
    )

    st.markdown(
        "### Calculating Reconstruction Errors and Setting Anomaly Thresholds")
    st.markdown(
        "The reconstruction error $e_i$ for a given input $x_i$ is calculated as the mean squared error (MSE) "
        "between the original input and its reconstruction $\hat{x}_i$:"
    )
    st.markdown(
        r"""
$$
e_i = \frac{1}{F} \sum_{j=1}^{F} (x_{ij} - \hat{x}_{ij})^2
$$""")
    st.markdown(
        r"where $F$ is the number of features, $x_{ij}$ is the $j$-th feature of the $i$-th observation, "
        r"and $\hat{x}_{ij}$ is its reconstructed value."
    )

    st.markdown(
        "To classify an observation as anomalous, its reconstruction error $e_i$ must exceed a threshold $\\tau$:")
    st.markdown(
        r"""
$$
\text{Anomaly}_i = \begin{cases} 1 & \text{if } e_i > \tau \\ 0 & \text{otherwise} \end{cases}
$$""")

    severity_legend()

    assumptions_box(
        [
            "Percentile thresholds are policy cutoffs, not probabilities of loss.",
            "Out-of-sample regimes can shift; alert rates may differ from in-sample percentiles.",
            "Anomaly can reflect unusual co-movement patterns even if headline returns are modest."
        ]
    )

    if st.session_state.autoencoder is None or st.session_state.X_train is None:
        st.warning(
            "Train the baseline model first on ‘Learn Normal Market Structure’.")
    else:
        percentiles_selected = st.multiselect(
            "Choose alert severity tiers (percentiles on training-period errors)",
            options=[90, 95, 99, 99.5],
            default=[95, 99],
            help="95% ≈ watchlist tier; 99% ≈ escalation tier. Choose tiers consistent with your triage bandwidth.",
        )

        st.markdown("### Checkpoint question (for intuition)")
        ans2 = st.radio(
            "If you move from a 95th to a 99th percentile threshold, what happens to alert frequency?",
            ["More alerts (lower bar)",
             "Fewer alerts (higher bar)", "No change"],
            index=1,
        )
        if ans2 == "Fewer alerts (higher bar)":
            st.success(
                "Correct. Higher percentiles are stricter and produce fewer alerts on the training distribution.")
        elif ans2:
            st.info(
                "Not quite. A 99th percentile threshold is stricter and typically yields fewer alerts than 95th.")

        if st.button("Compute anomaly scores & thresholds"):
            if not percentiles_selected:
                st.error("Select at least one percentile threshold.")
            else:
                with st.spinner("Computing reconstruction errors and thresholds..."):
                    st.session_state.train_recon_errors = calculate_reconstruction_errors(
                        st.session_state.autoencoder, st.session_state.X_train
                    )
                    st.session_state.test_recon_errors = calculate_reconstruction_errors(
                        st.session_state.autoencoder, st.session_state.X_test
                    )
                    st.session_state.anomaly_thresholds = set_anomaly_thresholds(
                        st.session_state.train_recon_errors, percentiles=percentiles_selected
                    )

                    test_dates = st.session_state.market_returns.index[st.session_state.test_mask]
                    anomaly_results_df = pd.DataFrame(
                        {
                            "date": test_dates,
                            "recon_error": st.session_state.test_recon_errors,
                            "SP500_ret": st.session_state.market_returns.loc[st.session_state.test_mask, "SP500"].values,
                            "VIX": st.session_state.market_returns.loc[st.session_state.test_mask, "VIX"].values,
                        }
                    ).set_index("date")

                    for p in percentiles_selected:
                        tau = st.session_state.anomaly_thresholds.get(p, None)
                        if tau is not None:
                            anomaly_results_df[f"flag_{p}pctl"] = anomaly_results_df["recon_error"] > tau

                    st.session_state.anomaly_results_df = anomaly_results_df
                    st.session_state.top_anomalies_df = anomaly_results_df.nlargest(
                        10, "recon_error")

                st.success(
                    "Alerts generated. Next: explain the alerts and map regimes.")

        if st.session_state.anomaly_results_df is not None:
            provenance_box(
                [
                    "Anomaly score = mean squared reconstruction error across F features on standardized inputs.",
                    "Thresholds = selected percentiles of **training-period** reconstruction errors.",
                    "Top anomalies are ranked by reconstruction error in the **test/surveillance** window."
                ]
            )

            st.markdown(
                "### Reconstruction error distribution (defines ‘normal’ in error-space)")
            plot_error_distribution(
                st.session_state.train_recon_errors, st.session_state.anomaly_thresholds)
            display_and_clear_all_figures()

            st.markdown(
                "#### Expected alert rate (training distribution intuition)")
            rows = []
            for p in sorted(st.session_state.anomaly_thresholds.keys()):
                expected_rate = max(0.0, 1.0 - (p / 100.0))
                rows.append({"Threshold (percentile)": f"{p}",
                            "Implied tail rate (≈ alerts)": f"{expected_rate:.1%}"})
            st.dataframe(pd.DataFrame(rows))

            st.markdown(
                "### Reconstruction error time series (surveillance timeline)")
            st.markdown(
                "Interpretation tip: isolated spikes suggest discrete events; a rising baseline can indicate a regime shift "
                "and may call for re-defining the ‘normal’ window or re-training governance policy."
            )
            plot_anomaly_time_series(
                st.session_state.anomaly_results_df,
                st.session_state.anomaly_thresholds,
                st.session_state.known_anomaly_dates,
            )
            display_and_clear_all_figures()

            st.markdown("### Top 10 anomalies (test window only)")
            cols = ["recon_error", "SP500_ret", "VIX"] + \
                [c for c in st.session_state.anomaly_results_df.columns if c.startswith(
                    "flag_")]
            st.dataframe(st.session_state.top_anomalies_df[cols])

            st.warning(
                "Guardrail: This table is **not** a forecast of returns and **not** a VaR exceedance list. "
                "It ranks days by inconsistency with learned cross-asset structure."
            )


# ============================================================
# Page 4: Anomaly Insights & Latent Space
# ============================================================
elif st.session_state.current_page == "Explain the Alert & Map Regimes":
    clear_matplotlib_cache()

    st.title("Explain the Alert: Drivers + Regime Map")

    st.markdown("## 6. Explain an alert (driver view)")
    st.markdown(
        "An alert is a triage signal, not a diagnosis. Here we break down **which features contributed most** "
        "to the reconstruction error on selected days."
    )

    st.markdown("### Per-Feature Error Analysis (Anomaly Gallery & Heatmap)")
    st.markdown(
        r"""
$$
\text{Per-Feature Error}_{ij} = (x_{ij} - \hat{x}_{ij})^2
$$""")

    st.markdown("### Watch-outs (common misconceptions)")
    st.markdown(
        "- A high anomaly score does **not** necessarily mean negative returns.\n"
        "- A low anomaly score does **not** necessarily mean low risk.\n"
        "- Anomalies often reflect **unusual co-movement**, not just large single-asset moves."
    )
    st.markdown("**Practitioner Warning:**")
    st.markdown(r"Anomaly $\neq$ Crisis.")

    assumptions_box(
        [
            "Per-feature errors are in standardized units; interpret relative magnitudes within the same scaling scheme.",
            "Driver attribution is ‘model space’ inconsistency, not causal explanation.",
            "Use this view to prioritize investigation: macro shock vs idiosyncratic sector event vs data issue."
        ]
    )

    if st.session_state.anomaly_results_df is None or st.session_state.autoencoder is None:
        st.warning("Generate alerts first on ‘Turn Deviation into Alerts’.")
    else:
        top_idx = st.session_state.top_anomalies_df.index if st.session_state.top_anomalies_df is not None else []
        options = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in top_idx]
        default = options[: min(5, len(options))] if options else []

        anomaly_dates_for_analysis = st.multiselect(
            "Pick alert dates to explain (from top anomalies)",
            options=options,
            default=default,
            help="Start with the top anomalies, then branch to dates you care about (e.g., event days).",
        )

        if st.button("Explain selected alerts (driver heatmap & gallery)"):
            if anomaly_dates_for_analysis:
                with st.spinner("Generating driver views..."):
                    analyze_per_feature_errors(
                        st.session_state.autoencoder,
                        st.session_state.X_test,
                        pd.to_datetime(anomaly_dates_for_analysis),
                        st.session_state.ae_features,
                    )
                    st.markdown(
                        "### Driver views (what moved inconsistently with normal structure)")
                    display_and_clear_all_figures()
            else:
                st.info("Select at least one date to explain.")

        st.info(
            "Micro-example: If errors concentrate in rates + USD features, investigate macro shock and cross-asset hedges; "
            "if concentrated in a single sector feature, investigate idiosyncratic news/liquidity."
        )

    st.markdown("---")
    st.markdown("## 7. Map regimes (latent space view)")
    st.markdown(
        "A 2D latent map is a regime lens: days that land near each other have similar cross-asset structure under the model. "
        "Coloring by a stress proxy (e.g., VIX change) helps interpret whether the learned representation aligns with intuition."
    )

    assumptions_box(
        [
            "Axes are not directly interpretable economic factors; treat the map as a similarity space.",
            "Clustering of stress periods suggests learned regime structure; a random cloud suggests weak regime signal.",
        ]
    )

    if st.session_state.X_train is None or st.session_state.X_test is None:
        st.warning("Prepare data first on ‘Build the Market Lens’.")
    else:
        if st.button("Map market regimes (2D latent space)"):
            with st.spinner("Building and visualizing regime map..."):
                build_and_visualize_latent_space(
                    st.session_state.X_train,
                    st.session_state.X_test,
                    st.session_state.market_returns,
                    st.session_state.train_mask,
                    st.session_state.test_mask,
                    st.session_state.ae_features,
                    st.session_state.anomaly_results_df,
                )
                display_and_clear_all_figures()

        st.markdown("### Decision translation (how to use this map)")
        st.markdown(
            "- If stress days cluster tightly, consider regime-conditioned risk monitoring (different limits/checks by regime).\n"
            "- If alerts drift into a new region over time, consider that a **structural regime shift** and revisit the training cutoff."
        )


# ============================================================
# Page 5: Model Benchmarking & VAE Extension
# ============================================================
elif st.session_state.current_page == "Validate the Alarm & Scenario Thinking":
    clear_matplotlib_cache()

    st.title("Validate the Alarm + Extend to Scenario Thinking")

    st.markdown(
        "## 8. Benchmark credibility: compare to standard surveillance baselines")
    st.markdown(
        "To build trust, we compare the alerting behavior against two standard baselines:\n"
        "- **Isolation Forest** (tree-based anomaly baseline)\n"
        "- **Mahalanobis Distance** (multivariate statistical distance)\n\n"
        "We use reference dates as imperfect labels; treat this as **directional benchmarking**, not definitive ground truth."
    )

    st.markdown("### Implementing and Comparing Baseline Models")
    st.markdown(
        "**Isolation Forest:** isolates anomalies via random splits. The anomaly score $s(x)$ is:")
    st.markdown(r"""
$$
s(x) = 2^{-\frac{E[h(x)]}{c(n)}}
$$""")
    st.markdown(
        r"where $E[h(x)]$ is the average path length to isolate $x$, and $c(n)$ is the expected path length for dataset size $n$."
    )

    st.markdown(
        "**Mahalanobis Distance:** measures how many 'standard deviations' away a point is in multivariate space:")
    st.markdown(r"""
$$
D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}
$$""")
    st.markdown(
        r"where $x$ is the data point, $\mu$ the mean vector, and $\Sigma^{-1}$ the inverse covariance matrix."
    )

    assumptions_box(
        [
            "Reference event dates are incomplete labels; PR curves measure agreement with reference list, not truth.",
            "Baselines assume different structure (linear covariance vs nonlinear co-movement); interpret comparisons accordingly.",
        ]
    )

    if st.session_state.autoencoder is None or st.session_state.anomaly_results_df is None:
        st.warning("Train the model and generate alerts before benchmarking.")
    else:
        if st.button("Benchmark vs baselines (credibility check)"):
            with st.spinner("Running baseline comparisons..."):
                # Capture print output from the function
                output_buffer = StringIO()
                with redirect_stdout(output_buffer):
                    compare_anomaly_detectors(
                        st.session_state.X_train,
                        st.session_state.X_test,
                        st.session_state.anomaly_results_df,
                        st.session_state.known_anomaly_dates,
                        st.session_state.test_recon_errors,
                    )
                
                # Display captured output
                output_text = output_buffer.getvalue()
                if output_text.strip():
                    st.text(output_text)
                
                # Display any figures created
                num_figs = display_and_clear_all_figures()
                if num_figs == 0:
                    st.info("No plots were generated. This may occur if there are no known anomaly dates in the test set.")

        st.markdown("### Decision translation (how to use benchmark results)")
        st.markdown(
            "- If the flexible model only marginally improves over Mahalanobis, governance may favor the simpler baseline.\n"
            "- If the flexible model meaningfully improves detection of reference stress events, it supports using it for escalation workflows."
        )


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
