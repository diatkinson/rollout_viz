import html
import json
import math
import re
from typing import Callable

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Precompile regex for performance
_COMPILED_NEWLINE_RE = re.compile(r"\n{2,}")

########################
# Helper functions
########################

def escape(s: str) -> str:
    """Escape special characters and newlines using a precompiled regex."""
    escaped_s = _COMPILED_NEWLINE_RE.sub("\n", html.escape(s)).replace("\n", "\\n")
    return escaped_s

def format_annotations(labels):
    labels = [escape(label) for label in labels]
    if len(labels) > 1:
        return "Multiple annotations:\n• " + "\n• ".join(labels)
    else:
        return labels[0]

def normalize_data(data, method):
    """Normalize data using specified method"""
    data = np.array(data, dtype=float)
    if method == "Min-max":
        if np.max(data) == np.min(data):
            return np.ones_like(data) * 0.5
        else:
            return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == "Z-score":
        if np.std(data) == 0:
            return np.zeros_like(data)
        else:
            return (data - np.mean(data)) / np.std(data)
    return data

@st.cache_data
def load_jsonl(file):
    """
    Read a text buffer containing JSONL data.
    Each line in the file is a JSON object with "tokens" and "metrics".
    Returns a list of parsed lines.
    """
    lines = []
    for line in file:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            lines.append(data)
        except json.JSONDecodeError:
            st.warning(f"Skipping malformed JSON line: {line[:50]}...")
    return lines

def formatted_next_tokens(next_tokens, label_name, val, num_top_tokens=5, new_line_token="\n"):
    base_str = f"{label_name}: {val:.3f}"
    if next_tokens is None:
        return base_str

    top_tokens = [
        (token, prob)
        for token, prob in list(sorted(next_tokens.items(), key=lambda x: x[1], reverse=True))[:num_top_tokens]
    ]
    max_token_len = max(len(token) for token, _ in top_tokens)
    next_tokens_str = new_line_token.join([f"{token:<{max_token_len}} {prob:.3f}" for token, prob in top_tokens])
    return f"{base_str}{new_line_token}----{new_line_token}{next_tokens_str}"

def color_tokens(
    tokens,
    values,
    metric_name,
    normalization_method,
    next_tokens=None,
    token_borders=False,
    annotations=None,
):
    """
    Create HTML spans with background color for each token based on the corresponding value.
    Optimized using NumPy vectorized operations.
    """
    tooltip_style = f"""
    <div>
    <style>
    .token-container .token-span {{
        position: relative;
        display: inline-block;
        padding: 0px 0px;
        margin: 1px 0; 
        border-radius: 2px;
        box-sizing: border-box;
        border: 0.5px solid {"rgba(0, 0, 0, 0.3)" if token_borders else "transparent"};
        z-index: 0;
    }}
    .token-container .span-annotation {{
        border-bottom: 3px solid #8A2BE2;
        padding-bottom: 4px;
        position: relative;
    }}
    .token-container .token-span:hover {{
        border-color: rgba(0, 0, 0, 0.2);
    }}
    .token-container .token-span:hover::after {{
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        padding: 5px 10px;
        background: rgba(0, 0, 0, 0.7);
        color: white;
        border-radius: 4px;
        font-size: 14px;
        white-space: pre;
        max-width: 1000px;
        z-index: 10;
        font-family: monospace;
    }}
    .token-container .span-annotation:hover::after {{
        content: attr(data-tooltip);
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        padding: 5px 10px;
        background: rgba(0, 0, 0, 0.7);
        color: white;
        border-radius: 4px;
        font-size: 14px;
        white-space: pre;
        max-width: 1000px;
        z-index: 1000;
        font-family: monospace;
    }}
    </style>
    <div class="token-container">
    """

    next_tokens_lst = next_tokens or [None] * len(tokens)
    min_val = min(values)
    max_val = max(values)
    mean_val = sum(values) / len(values)
    max_deviation = max(abs(max_val - mean_val), abs(min_val - mean_val))
    max_abs_val = max(abs(max_val), abs(min_val))
    arr = np.array(values, dtype=float)

    if normalization_method == "White to Green":
        if max_val == min_val:
            scale = np.full_like(arr, 0.5, dtype=float)
        else:
            scale = (arr - min_val) / (max_val - min_val)
        red_intensities = (255 - (255 * scale)).astype(int)
        blue_intensities = (255 - (255 * scale)).astype(int)
        green_intensities = np.full(arr.shape, 255, dtype=int)
    elif normalization_method == "Red to Green":
        if max_deviation == 0:
            scale = np.zeros_like(arr, dtype=float)
        else:
            scale = (arr - mean_val) / max_deviation
        red_intensities = np.where(scale <= 0, 255, (255 * (1 - scale)).astype(int))
        blue_intensities = np.where(scale <= 0, (255 * (1 + scale)).astype(int), (255 * (1 - scale)).astype(int))
        green_intensities = np.where(scale <= 0, (255 * (1 + scale)).astype(int), 255)
    elif normalization_method == "None":
        cap = max_abs_val if max_abs_val > 0 else 1.0
        scale = np.minimum(1.0, np.abs(arr) / cap)
        red_intensities = np.where(arr > 0, (255 * (1 - scale)).astype(int), 255)
        blue_intensities = np.where(arr > 0, (255 * (1 - scale)).astype(int),
                                    np.where(arr < 0, (255 * (1 - scale)).astype(int), 255))
        green_intensities = np.where(arr > 0, 255,
                                     np.where(arr < 0, (255 * (1 - scale)).astype(int), 255))
    else:
        red_intensities = np.full(arr.shape, 255, dtype=int)
        green_intensities = np.full(arr.shape, 255, dtype=int)
        blue_intensities = np.full(arr.shape, 255, dtype=int)

    colored_text = []
    for i, (token, nxt) in enumerate(zip(tokens, next_tokens_lst)):
        color_str = f"rgb({red_intensities[i]},{green_intensities[i]},{blue_intensities[i]})"
        tooltip_content = f"{values[i]:.3f}" if nxt is None else formatted_next_tokens(nxt, metric_name, values[i])
        span = f'<span class="token-span" style="background-color: {color_str};" data-tooltip="{tooltip_content}">{token}</span>'
        colored_text.append(span)

    # Process annotations if provided
    if annotations:
        token_annotations = [[] for _ in range(len(tokens))]
        for anno in annotations:
            start, end, label = anno["start"], anno["end"], anno["label"]
            for i in range(start, min(end, len(tokens))):
                token_annotations[i].append(label)

        spans_to_add = []
        current_start = None
        current_labels = None
        for i, labels in enumerate(token_annotations):
            labels_sorted = sorted(labels)
            if labels_sorted != current_labels:
                if current_labels is not None and current_start is not None:
                    spans_to_add.append((current_start, i, format_annotations(current_labels)))
                if labels:
                    current_start = i
                    current_labels = labels_sorted
                else:
                    current_start = None
                    current_labels = None
        if current_labels is not None and current_start is not None:
            spans_to_add.append((current_start, len(tokens), format_annotations(current_labels)))
        # Replace segments in colored_text with annotation-wrapped HTML
        for start, end, label in spans_to_add:
            tokens_html = " ".join(colored_text[start:end])
            span_html = f'<span class="span-annotation" data-tooltip="{label}">{tokens_html}</span>'
            colored_text[start:end] = [span_html]
        result = tooltip_style + " ".join(colored_text)
    else:
        result = tooltip_style + " ".join(colored_text)

    result += "</div></div>"
    return result

def create_token_plot(
    tokens: list[str],
    metrics: dict[str, list[float]],
    next_tokens: list[dict[str, float]] | None = None,
    normalization_method: Callable[[list[float]], list[float]] = lambda x: x,
    tokens_per_line: int = 10,
    use_scattergl: bool = True,
    combine_subplots: bool = False,
) -> go.Figure:
    """
    Creates a Plotly figure displaying token metrics.
    Optimizations:
      - Option to use go.Scattergl for WebGL-based rendering.
      - Option to combine all tokens into a single subplot to reduce memory usage.
    """
    next_tokens = next_tokens or [None] * len(tokens)
    available_colors = px.colors.qualitative.Plotly
    metric_names = list(metrics.keys())
    color_dict = {m: available_colors[i % len(available_colors)] for i, m in enumerate(metric_names)}

    if combine_subplots:
        fig = go.Figure()
        for metric_name, metric_values in metrics.items():
            normalized_values = normalization_method(metric_values)
            hover_text = [
                (
                    "<span style='font-family: monospace;'>"
                    + formatted_next_tokens(nt, token, raw_val, new_line_token="<br>")
                    + "</span>"
                )
                for token, nt, raw_val in zip(tokens, next_tokens, metric_values)
            ]
            trace_type = go.Scattergl if use_scattergl else go.Scatter
            fig.add_trace(trace_type(
                x=list(range(len(tokens))),
                y=normalized_values,
                mode="lines+markers",
                name=metric_name,
                line=dict(color=color_dict[metric_name]),
                hovertext=hover_text,
                hoverinfo="text",
            ))
        fig.update_layout(
            height=600,
            title="Token Metrics Plot",
            xaxis=dict(tickmode="array", tickvals=list(range(len(tokens))), ticktext=tokens),
            hovermode="closest",
            margin=dict(t=30, b=10, l=10, r=10),
        )
        return fig
    else:
        num_chunks = math.ceil(len(tokens) / tokens_per_line)
        fig = make_subplots(rows=num_chunks, cols=1, shared_xaxes=False, shared_yaxes="all")
        for chunk_index in range(num_chunks):
            start_i = chunk_index * tokens_per_line
            end_i = start_i + tokens_per_line
            chunk_tokens = tokens[start_i:end_i]
            for metric_name, metric_values in metrics.items():
                chunk_values = metric_values[start_i:end_i]
                normalized_chunk_values = normalization_method(chunk_values)
                chunk_next_tokens = next_tokens[start_i:end_i]
                hover_text = [
                    (
                        "<span style='font-family: monospace;'>"
                        + formatted_next_tokens(nt, token, raw_val, new_line_token="<br>")
                        + "</span>"
                    )
                    for token, nt, raw_val in zip(chunk_tokens, chunk_next_tokens, chunk_values)
                ]
                trace_type = go.Scattergl if use_scattergl else go.Scatter
                fig.add_trace(trace_type(
                    x=list(range(len(chunk_tokens))),
                    y=normalized_chunk_values,
                    mode="lines+markers",
                    name=metric_name,
                    legendgroup=metric_name,
                    showlegend=(chunk_index == 0),
                    line=dict(color=color_dict[metric_name]),
                    hovertext=hover_text,
                    hoverinfo="text",
                ), row=chunk_index + 1, col=1)
            fig.update_xaxes(
                tickmode="array",
                tickvals=list(range(len(chunk_tokens))),
                ticktext=chunk_tokens,
                row=chunk_index + 1,
                col=1,
                tickfont=dict(size=12),
            )
        fig.update_layout(
            height=250 * num_chunks,
            showlegend=True,
            title="Token Metrics Plot",
            margin=dict(t=30, b=10, l=10, r=10),
            hovermode="closest",
        )
        return fig

########################
# Streamlit interface
########################

st.set_page_config(layout="wide")

st.title("Rollout Metrics")

st.markdown(
    """
    We expect a JSONL file where each line is a JSON object with the following keys:
    - `tokens`: list of tokens (strings)
    - `metrics`: dictionary of metric_name -> list of float values (same length as tokens)
    - `next_tokens` (optional): list of dictionaries (same length as tokens), which each map from a possible next token to its associated probability (or logits)
    - `annotations` (optional): list of span annotations, each with:
        - `start`: starting token index
        - `end`: ending token index (exclusive)
        - `label`: annotation text
    """
)

uploaded_file = st.file_uploader("Upload your JSONL file", type=["jsonl"])

if "data" not in st.session_state:
    st.session_state["data"] = []

if uploaded_file is not None:
    st.session_state["data"] = load_jsonl(uploaded_file)

data = st.session_state["data"]

if data:
    col1, col2 = st.columns(2)

    with col1:
        line_index = st.number_input("Line index to display", min_value=0, max_value=len(data) - 1, value=0)
        display_mode = st.radio("Display mode", ["Text Color", "Line Plot (slow)"])

    current_line = data[line_index]

    tokens = current_line["tokens"]
    metrics = {m: vs for m, vs in current_line["metrics"].items()}

    tokens = [escape(token) for token in tokens]
    if current_line.get("next_tokens"):
        current_line["next_tokens"] = [
            {escape(token): prob for token, prob in next_tokens.items()}
            for next_tokens in current_line["next_tokens"]
        ]

    if display_mode == "Text Color":
        metric_list = list(metrics.keys())
        if metric_list:
            with col1:
                normalization_method = st.radio("Color Normalization", ["None", "White to Green", "Red to Green"])
            with col2:
                selected_metric = st.radio("Choose a metric to color by", [f"`{m}`" for m in metric_list])
                token_borders = st.checkbox("Show token borders", value=False)
                if "annotations" in current_line:
                    show_annotations = st.checkbox("Show annotations", value=True)
                else:
                    show_annotations = False
            metric_values = metrics[selected_metric.strip("`")]
            colored_html = color_tokens(
                tokens,
                metric_values,
                metric_name=selected_metric.strip("`"),
                normalization_method=normalization_method,
                next_tokens=current_line.get("next_tokens"),
                token_borders=token_borders,
                annotations=current_line.get("annotations") if show_annotations else None,
            )
            st.markdown(colored_html, unsafe_allow_html=True)
        else:
            with col2:
                st.warning("No metrics found in this file.")
    else:
        with col1:
            normalization_method = st.radio("Normalization", ["None", "Min-max", "Z-score"])
            tokens_per_line = st.number_input("Tokens per line", min_value=3, value=20)
            use_scattergl = st.checkbox("Use WebGL (Scattergl)", value=True)
            combine_subplots = st.checkbox("Combine subplots into a single plot", value=False)

        metric_list = list(metrics.keys())
        selected_metrics = []
        with col2:
            st.write("Metrics to plot:")
            for m in metric_list:
                if st.checkbox(f"{m}", value=True):
                    selected_metrics.append(m)

        fig = create_token_plot(
            tokens=tokens,
            metrics={k: v for k, v in metrics.items() if k in selected_metrics},
            normalization_method=lambda x: normalize_data(x, normalization_method),
            next_tokens=current_line.get("next_tokens"),
            tokens_per_line=tokens_per_line,
            use_scattergl=use_scattergl,
            combine_subplots=combine_subplots,
        )
        st.plotly_chart(fig)

else:
    st.info("Please upload a JSONL file to proceed.")
