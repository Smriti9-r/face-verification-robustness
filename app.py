
from pathlib import Path
import csv

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
METRICS_FILE = RESULTS / "combined_heldout_robustness.csv"

st.set_page_config(
    page_title="Face Verification Robustness Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.18);
            padding: 16px;
            border-radius: 12px;
        }

        .research-note {
            padding: 14px 18px;
            border-radius: 10px;
            background: rgba(70, 130, 180, 0.12);
            border-left: 4px solid #4682B4;
            margin: 12px 0 20px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD SAVED EXPERIMENTAL RESULTS
# ============================================================

@st.cache_data
def load_metrics(path, modified_time):
    with open(path, newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


if not METRICS_FILE.exists():
    st.error(
        "Results file not found:\n\n"
        f"{METRICS_FILE}"
    )
    st.stop()

try:
    rows = load_metrics(
        str(METRICS_FILE),
        METRICS_FILE.stat().st_mtime,
    )
except Exception as error:
    st.error(f"Could not load the results CSV: {error}")
    st.stop()


REQUIRED_COLUMNS = {
    "condition",
    "severity",
    "auc",
    "eer",
    "far_at_validation_threshold",
    "frr_at_validation_threshold",
}

if not rows:
    st.error("The results CSV contains no data.")
    st.stop()

missing_columns = REQUIRED_COLUMNS - set(rows[0].keys())

if missing_columns:
    st.error(
        "Missing columns in results CSV: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()

try:
    for row in rows:
        row["condition"] = row["condition"].strip().lower()

        for column in REQUIRED_COLUMNS - {"condition"}:
            row[column] = float(row[column])

except (ValueError, TypeError, KeyError) as error:
    st.error(f"Invalid data in the results CSV: {error}")
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_condition_rows(condition):
    return sorted(
        [r for r in rows if r["condition"] == condition],
        key=lambda r: r["severity"],
    )


def find_baseline():
    for row in rows:
        if row["condition"] == "blur" and row["severity"] == 0:
            return row
    return None


def show_metrics(row):
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("AUC", f'{row["auc"]:.4f}')
    c2.metric("EER", f'{row["eer"]:.2%}')
    c3.metric(
        "FAR at fixed threshold",
        f'{row["far_at_validation_threshold"]:.2%}',
    )
    c4.metric(
        "FRR at fixed threshold",
        f'{row["frr_at_validation_threshold"]:.2%}',
    )


def make_chart(data, metric, title, ylabel, condition):
    x_values = [r["severity"] for r in data]
    y_values = [r[metric] for r in data]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        x_values,
        y_values,
        marker="o",
        linewidth=2,
        markersize=6,
    )

    ax.set_title(title)
    ax.set_xlabel(
        "Blur radius" if condition == "blur"
        else "Brightness factor"
    )
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    if metric != "auc":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    fig.tight_layout()
    return fig


def show_table(data):
    st.dataframe(
        [
            {
                "Severity": r["severity"],
                "AUC": round(r["auc"], 4),
                "EER": f'{r["eer"]:.2%}',
                "FAR": f'{r["far_at_validation_threshold"]:.2%}',
                "FRR": f'{r["frr_at_validation_threshold"]:.2%}',
            }
            for r in data
        ],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔬 Research Dashboard")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Robustness Lab", "Results & Analysis"],
)

st.sidebar.divider()
st.sidebar.markdown("**Research details**")
st.sidebar.caption("Dataset: Labeled Faces in the Wild")
st.sidebar.caption("Model: InceptionResnetV1")
st.sidebar.caption("Embedding dimension: 512")
st.sidebar.caption("Similarity: Cosine similarity")

st.sidebar.divider()
st.sidebar.caption(
    "Educational research prototype. "
    "Not intended for real-world identity decisions."
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("Face Verification Robustness Lab")
st.caption(
    "Evaluating how image degradation affects a pretrained "
    "face verification model."
)


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Project Overview")

    st.write(
        "This project investigates how Gaussian blur and reduced "
        "brightness affect face verification using pretrained facial "
        "embeddings and cosine similarity."
    )

    st.markdown(
        """
        <div class="research-note">
        This dashboard displays previously computed experiments.
        It does not retrain the model or rerun the experiments.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Experimental setup")

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset", "LFW")
    c2.metric("Embedding size", "512")
    c3.metric("Model", "InceptionResnetV1")

    st.subheader("Verification protocol")

    st.markdown(
        """
        - **Genuine pairs:** Images of the same identity.
        - **Impostor pairs:** Images of different identities.
        - **Threshold selection:** Selected using the validation set.
        - **Held-out testing:** Test identities are separate from
          validation identities.
        - **Degradation:** The second image in each pair is transformed,
          while the first remains unchanged.
        """
    )

    st.subheader("Baseline performance")

    baseline = find_baseline()

    if baseline:
        show_metrics(baseline)
        st.caption(
            "Metrics are taken from the undegraded baseline row."
        )
    else:
        st.warning("The baseline row was not found.")

    st.subheader("Research questions")

    st.markdown(
        """
        1. How does increasing blur affect verification?
        2. How does reducing brightness affect verification?
        3. How do FAR and FRR change under image degradation?
        4. Which degradation has the larger observed effect?
        """
    )

    st.subheader("Research scope")

    st.write(
        "This is a robustness study under controlled image degradation. "
        "Demographic fairness was not evaluated."
    )


# ============================================================
# PAGE 2: ROBUSTNESS LAB
# ============================================================

elif page == "Robustness Lab":

    st.header("Robustness Lab")

    st.write(
        "Select a degradation and explore the saved verification results."
    )

    condition = st.selectbox(
        "Image degradation",
        ["blur", "brightness"],
        format_func=lambda value: (
            "Gaussian blur" if value == "blur"
            else "Reduced brightness"
        ),
    )

    subset = get_condition_rows(condition)

    if not subset:
        st.warning("No results were found for this degradation.")
        st.stop()

    severity_values = [r["severity"] for r in subset]

    # Clickable options instead of a difficult-to-control slider.
    selected_severity = st.radio(
        "Select severity",
        options=severity_values,
        index=0,
        horizontal=True,
        format_func=lambda value: f"{value:g}",
        help=(
            "Click a tested severity level. "
            "The saved metrics update without rerunning the model."
        ),
    )

    selected = next(
        r for r in subset
        if r["severity"] == selected_severity
    )

    if condition == "blur":
        st.caption("Higher blur radius means stronger Gaussian blur.")
    else:
        st.caption(
            "Lower brightness factor means a darker image; "
            "1.0 represents the original brightness."
        )

    st.subheader(f"Metrics at severity {selected_severity:g}")

    show_metrics(selected)

    st.divider()

    st.subheader("Performance across severity levels")

    left, right = st.columns(2)

    with left:
        fig_auc = make_chart(
            subset,
            "auc",
            "AUC across severity levels",
            "AUC",
            condition,
        )
        st.pyplot(fig_auc, use_container_width=True)
        plt.close(fig_auc)

    with right:
        fig_frr = make_chart(
            subset,
            "frr_at_validation_threshold",
            "False rejection rate",
            "FRR",
            condition,
        )
        st.pyplot(fig_frr, use_container_width=True)
        plt.close(fig_frr)

    st.subheader("All severity levels")
    show_table(subset)

    st.divider()

    st.subheader("Understanding the metrics")

    with st.expander("AUC — ranking performance"):
        st.write(
            "AUC measures how well genuine pairs tend to receive higher "
            "similarity scores than impostor pairs. AUC near 1 indicates "
            "strong ranking performance; near 0.5 indicates near-chance "
            "ranking."
        )

    with st.expander("EER — equal error rate"):
        st.write(
            "EER is the point at which false acceptance and false "
            "rejection rates are equal or approximately equal. "
            "Lower EER generally indicates better separation."
        )

    with st.expander("FAR — false acceptance rate"):
        st.write(
            "FAR is the proportion of impostor pairs incorrectly "
            "accepted at the fixed validation threshold."
        )

    with st.expander("FRR — false rejection rate"):
        st.write(
            "FRR is the proportion of genuine pairs incorrectly "
            "rejected at the fixed validation threshold."
        )

    st.info(
        "FAR and FRR use the fixed validation threshold. "
        "EER is calculated separately from the score distribution."
    )


# ============================================================
# PAGE 3: RESULTS & ANALYSIS
# ============================================================

elif page == "Results & Analysis":

    st.header("Results & Analysis")

    st.write(
        "These saved figures summarize held-out performance under "
        "blur and brightness degradation."
    )

    auc_path = FIGURES / "heldout_auc.png"
    frr_path = FIGURES / "heldout_frr.png"

    if auc_path.exists():
        st.subheader("AUC under image degradation")
        st.image(
            str(auc_path),
            caption="Held-out AUC under image degradation.",
            use_container_width=True,
        )
    else:
        st.warning("The held-out AUC figure was not found.")

    if frr_path.exists():
        st.subheader("False rejection rate under image degradation")
        st.image(
            str(frr_path),
            caption="FRR at the fixed validation threshold.",
            use_container_width=True,
        )
    else:
        st.warning("The held-out FRR figure was not found.")

    st.subheader("Compare degradation metrics")

    blur_rows = get_condition_rows("blur")
    brightness_rows = get_condition_rows("brightness")

    if blur_rows and brightness_rows:

        selected_metric = st.radio(
            "Metric",
            ["AUC", "EER", "FAR", "FRR"],
            horizontal=True,
        )

        metric_map = {
            "AUC": ("auc", "AUC"),
            "EER": ("eer", "EER"),
            "FAR": (
                "far_at_validation_threshold",
                "FAR at fixed threshold",
            ),
            "FRR": (
                "frr_at_validation_threshold",
                "FRR at fixed threshold",
            ),
        }

        metric_key, metric_label = metric_map[selected_metric]

        # Separate panels avoid suggesting that blur radius and
        # brightness factor share the same severity scale.
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for ax, data, condition_name, label in [
            (axes[0], brightness_rows, "brightness", "Brightness"),
            (axes[1], blur_rows, "blur", "Gaussian blur"),
        ]:
            x_values = [r["severity"] for r in data]
            y_values = [r[metric_key] for r in data]

            ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=2,
            )

            ax.set_title(label)
            ax.set_xlabel(
                "Brightness factor" if condition_name == "brightness"
                else "Blur radius"
            )
            ax.set_ylabel(metric_label)
            ax.grid(True, alpha=0.3)

            if metric_key == "auc":
                ax.set_ylim(0, 1.05)
            else:
                ax.set_ylim(0, 1.05)
                ax.yaxis.set_major_formatter(
                    PercentFormatter(xmax=1)
                )

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    else:
        st.warning(
            "Both blur and brightness data are needed for comparison."
        )

    st.subheader("Key findings")

    st.markdown(
        """
        - Baseline performance is strong on the selected held-out pairs.
        - Moderate brightness reductions have relatively little effect.
        - Severe darkening reduces verification performance.
        - Increasing blur causes substantial performance deterioration.
        - Under severe blur, AUC approaches 0.5 and FRR becomes very high.
        """
    )

    st.subheader("Limitations")

    st.markdown(
        """
        - Only one pretrained model and one dataset were evaluated.
        - Controlled transformations do not represent every real-world
          imaging condition.
        - Only the second image in each pair was degraded.
        - Images may be reused across pairs, so pair observations are
          not fully independent.
        - The preliminary bootstrap interval resamples scores rather
          than identity clusters.
        - Demographic fairness was not evaluated.
        """
    )

    report_path = ROOT / "REPORT.md"

    st.subheader("Research report")

    if report_path.exists():
        st.success("REPORT.md is present in the project folder.")
        st.caption(
            "Open REPORT.md in VS Code to read the complete report."
        )
    else:
        st.warning("REPORT.md was not found.")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Face Verification Robustness Lab | Educational research prototype "
    "| Not intended for real-world identity decisions"
)
