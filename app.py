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

########################
# Helper functions
########################
_COMPILED_NEWLINE_RE = re.compile(r"\n{2,}")
#pre-compile to save time
def escape(s: str) -> str:
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


#cache json loading
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

    colored_text = []
    for token, val, next_tokens in zip(tokens, values, next_tokens_lst):
        # Normalize values according to the method
        if normalization_method == "White to Green":
            # Scale from 0 to 255 for green intensity
            scale = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            red_intensity = 255 - int(255 * scale)  # Fade from 255 to 0
            green_intensity = 255  # Keep green at max
            blue_intensity = 255 - int(255 * scale)  # Fade from 255 to 0
        elif normalization_method == "Red to Green":
            # Scale from -1 to 1 centered at mean
            if max_deviation == 0:
                scale = 0
            else:
                scale = (val - mean_val) / max_deviation

            if scale <= 0:  # Red to White
                red_intensity = 255
                green_intensity = blue_intensity = int(255 * (1 + scale))
            else:  # White to Green
                green_intensity = 255
                red_intensity = blue_intensity = int(255 * (1 - scale))
        else:
            # For unnormalized values: 0 is white, positive is green, negative is red
            # Find the maximum absolute value for scaling
            cap = max_abs_val if max_abs_val > 0 else 1.0

            if val == 0:
                red_intensity = green_intensity = blue_intensity = 255
            elif val > 0:
                # Scale from white to green
                scale = min(1.0, val / cap)  # Scale relative to max value
                green_intensity = 255
                red_intensity = blue_intensity = int(255 * (1 - scale))
            else:
                # Scale from white to red
                scale = min(1.0, abs(val) / cap)  # Scale relative to max value
                red_intensity = 255
                green_intensity = blue_intensity = int(255 * (1 - scale))

        color_str = f"rgb({red_intensity},{green_intensity},{blue_intensity})"

        # Create a more detailed tooltip content
        tooltip_content = f"{val:.3f}" if next_tokens is None else formatted_next_tokens(next_tokens, metric_name, val)
        span = f'<span class="token-span" style="background-color: {color_str};" data-tooltip="{tooltip_content}">{token}</span>'
        colored_text.append(span)

    result = tooltip_style + " ".join(colored_text)

    # Add span annotations if provided and enabled
    if annotations:
        # First, build a map of which tokens are covered by which annotations
        token_annotations = [[] for _ in range(len(tokens))]
        for anno in annotations:
            for i in range(anno["start"], anno["end"]):
                token_annotations[i].append(anno["label"])

        # Find continuous spans of tokens with the same set of annotations
        current_span = []
        current_labels = None
        spans_to_add = []

        for i, labels in enumerate(token_annotations):
            labels = sorted(labels)  # Sort labels for consistent comparison
            if labels != current_labels:
                if current_span and current_labels:  # Only add span if it had annotations
                    label_text = format_annotations(current_labels)

                    spans_to_add.append({"start": current_span[0], "end": current_span[-1] + 1, "label": label_text})
                current_span = [i] if labels else []  # Only start new span if there are labels
                current_labels = labels if labels else None
            elif labels:  # Only extend span if there are labels
                current_span.append(i)

        # Add the last span if it had annotations
        if current_span and current_labels:
            label_text = format_annotations(current_labels)

            spans_to_add.append({"start": current_span[0], "end": current_span[-1] + 1, "label": label_text})

        # Apply the combined spans
        for span in spans_to_add:
            start, end = span["start"], span["end"]
            label = span["label"]
            span_html = f'<span class="span-annotation" data-tooltip="{label}">'
            tokens_html = " ".join(colored_text[start:end])
            result = result.replace(tokens_html, f"{span_html}{tokens_html}</span>")

    result += "</div></div>"
    return result


def create_token_plot(
    tokens: list[str],
    metrics: dict[str, list[float]],
    next_tokens: list[dict[str, float]] | None = None,
    normalization_method: Callable[[list[float]], list[float]] = lambda x: x,
    tokens_per_line: int = 10,
) -> go.Figure:
    """
    Creates a plotly figure displaying tokens in groups (lines) of 'tokens_per_line',
    with each line showing a line plot of the provided metrics over that subset of tokens.
    Each metric has a unique color across all chunks, and the legend shows only one entry
    per metric.

    Args:
        tokens (List[str]): The list of tokens (strings).
        metrics (Dict[str, List[float]]): A dictionary of metric_name -> list of values
                                          (same length as tokens).
        normalization_method (Callable[[List[float]], List[float]]): A function
            that takes a list of float values and returns a list of normalized
            float values (same length).
        tokens_per_line (int, optional): Number of tokens to display per line (subplot).
                                         Defaults to 10.

    Returns:
        go.Figure: A Plotly figure with one row per chunk of tokens.
    """
    next_tokens = next_tokens or [None] * len(tokens)

    # Create a color dictionary for each metric using a Plotly palette (or any palette you like)
    available_colors = px.colors.qualitative.Plotly  # e.g. 10 distinct colors
    metric_names = list(metrics.keys())
    color_dict = {m: available_colors[i % len(available_colors)] for i, m in enumerate(metric_names)}

    # Number of chunks (plot rows) we'll display
    num_chunks = math.ceil(len(tokens) / tokens_per_line)

    fig = make_subplots(rows=num_chunks, cols=1, shared_xaxes=False, shared_yaxes="all")

    # Iterate over chunks
    for chunk_index in range(num_chunks):
        start_i = chunk_index * tokens_per_line
        end_i = start_i + tokens_per_line
        chunk_tokens = tokens[start_i:end_i]

        # For each metric, slice the corresponding chunk of values and optionally normalize
        for metric_name, metric_values in metrics.items():
            chunk_values = metric_values[start_i:end_i]
            normalized_chunk_values = normalization_method(chunk_values)
            chunk_next_tokens = next_tokens[start_i:end_i]

            # Create custom hover text combining token and value

            hover_text = [
                (
                    "<span style='font-family: monospace;'>"
                    + formatted_next_tokens(next_tokens, current_token, raw_value, new_line_token="<br>")
                    + "</span>"
                )
                for current_token, next_tokens, raw_value in zip(chunk_tokens, chunk_next_tokens, chunk_values)
            ]

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(chunk_tokens))),
                    y=normalized_chunk_values,
                    mode="lines+markers",
                    name=metric_name,
                    legendgroup=metric_name,
                    showlegend=(chunk_index == 0),
                    line=dict(color=color_dict[metric_name]),
                    hovertext=hover_text,
                    hoverinfo="text",
                ),
                row=chunk_index + 1,
                col=1,
            )

        # Update the x-axis for this row so tick labels show the actual tokens
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

# 1. Button to upload a JSONL file
uploaded_file = st.file_uploader("Upload your JSONL file", type=["jsonl"])

# Prepare a space to store the loaded data
if "data" not in st.session_state:
    st.session_state["data"] = []

if uploaded_file is not None:
    # Load the file and store it in session state
    st.session_state["data"] = load_jsonl(uploaded_file)

data = st.session_state["data"]

# If data is loaded, proceed
if data:
    col1, col2 = st.columns(2)

    with col1:
        line_index = st.number_input("Line index to display", min_value=0, max_value=len(data) - 1, value=0)
        display_mode = st.radio("Display mode", ["Text Color", "Line Plot (slow)"])

    # Retrieve the selected line data
    current_line = data[line_index]

    tokens = current_line["tokens"]
    metrics = {m: vs for m, vs in current_line["metrics"].items()}  # e.g. {"scoreA": [...], "scoreB": [...], ...}

    # Sanitize the tokens and next_tokens
    tokens = [escape(token) for token in tokens]
    if current_line.get("next_tokens"):
        current_line["next_tokens"] = [
            {escape(token): prob for token, prob in next_tokens.items()} for next_tokens in current_line["next_tokens"]
        ]

    if display_mode == "Text Color":
        # 4. If text color is chosen, let the user pick the metric
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
            # Retrieve the metric values for the chosen metric
            metric_values = metrics[selected_metric.strip("`")]

            # Generate HTML for colored tokens with annotations
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
        # 5. If line plot is chosen
        #    - Provide checkboxes to select which metrics to plot
        #    - Radio button for normalization method
        with col1:
            normalization_method = st.radio("Normalization", ["None", "Min-max", "Z-score"])
            tokens_per_line = st.number_input("Tokens per line", min_value=3, value=20)

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
        )
        st.plotly_chart(fig)

else:
    st.info("Please upload a JSONL file to proceed.")