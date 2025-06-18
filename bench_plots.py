import json
import os
import pandas as pd
import re
from bokeh.plotting import figure, show, output_file
from bokeh.models import (
    ColumnDataSource,
    NumeralTickFormatter,
    FactorRange,
)
from bokeh.palettes import Category10, Bright6  # Palettes for distinct colors
from bokeh.transform import factor_cmap

# --- Configuration ---
CRITERION_BASE_DIR = os.path.join("target", "criterion")
PLOT_OUTPUT_DIR = "benchmark_plots_bokeh"
BENCHMARK_GROUPS = ["Addition", "Cosine"]
GIB_CONVERSION_FACTOR = 1024 * 1024 * 1024

os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)


def parse_estimates(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data.get("mean", {}).get("point_estimate", None)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None


def format_bytes_for_label(bytes_val):
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024**2:
        return f"{bytes_val/1024:.0f} KiB"
    elif bytes_val < 1024**3:
        return f"{bytes_val/1024**2:.0f} MiB"
    else:
        return f"{bytes_val/1024**3:.0f} GiB"


def plot_line_chart_bokeh(group_name, df_group):
    if df_group.empty:
        print(f"No data to plot line chart for group: {group_name}")
        return

    df_group["Size"] = pd.to_numeric(df_group["Size"])
    df_group["Size_Bytes"] = df_group["Size"] * 4
    df_group["Throughput (GiB/s)"] = (
        df_group["Size_Bytes"] / (df_group["Time (ns)"] * 1e-9)
    ) / GIB_CONVERSION_FACTOR
    df_group["Time_ms"] = df_group["Time (ns)"] / 1_000_000
    df_group["Size_Label_Tooltip"] = df_group["Size_Bytes"].apply(
        format_bytes_for_label
    )

    output_filename = os.path.join(
        PLOT_OUTPUT_DIR, f"{group_name.replace(' ', '_')}_line_performance.html"
    )
    output_file(output_filename)

    tooltips = [
        ("Implementation", "@Implementation"),
        ("Vector Elements", "@Size{0,0}"),
        ("Vector Size", "@Size_Label_Tooltip"),
        ("Mean Time", "@Time_ms{0,0.000} ms"),
        ("Throughput", "@{Throughput (GiB/s)}{0,0.00} GiB/s"),
    ]

    p = figure(
        title=f"Performance Over Vector Size: {group_name}",
        x_axis_label="Vector Size (Number of f32 elements)",
        y_axis_label="Mean Time (milliseconds)",
        x_axis_type="log",
        y_axis_type="log",
        height=500,
        width=800,
        tooltips=tooltips,
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
    )

    p.xaxis.formatter = NumeralTickFormatter(format="0,0")
    p.yaxis.formatter = NumeralTickFormatter(format="0,0.000a")

    implementations = df_group["Implementation"].unique()
    palette = Category10.get(max(3, len(implementations)))

    for i, impl_name in enumerate(implementations):
        source_df = df_group[df_group["Implementation"] == impl_name].sort_values(
            by="Size"
        )
        source = ColumnDataSource(source_df)
        color = palette[i % len(palette)]
        p.line(
            x="Size",
            y="Time_ms",
            source=source,
            legend_label=impl_name,
            line_width=2,
            color=color,
        )
        p.scatter(
            x="Size",
            y="Time_ms",
            source=source,
            legend_label=impl_name,
            size=8,
            color=color,
            alpha=0.8,
        )

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    show(p)
    print(f"Bokeh line chart saved to {output_filename}")


def plot_grouped_bar_chart_per_size_bokeh(
    group_name, df_group_for_size, size_elements, size_label
):
    """Generates a bar chart for a single vector size, comparing implementations."""
    if df_group_for_size.empty:
        print(f"No data to plot bar chart for group: {group_name}, Size: {size_label}")
        return

    # Data is already filtered for a specific size
    # Implementations will be the categories on the x-axis
    implementations = df_group_for_size["Implementation"].unique().tolist()
    # Sort by time for potentially better visual order, or keep as is
    df_plot = df_group_for_size.sort_values(by="Time (ns)", ascending=True)
    implementations_sorted = df_plot[
        "Implementation"
    ].tolist()  # Use sorted order for x-axis

    source = ColumnDataSource(df_plot)

    output_filename = os.path.join(
        PLOT_OUTPUT_DIR,
        f"{group_name.replace(' ', '_')}_bar_size_{str(size_elements).replace(' ', '_')}.html",
    )
    output_file(output_filename)

    tooltips = [
        ("Implementation", "@Implementation"),
        ("Mean Time", "@Time_ms{0,0.000} ms"),
        ("Vector Elements", str(size_elements)),  # Add static size info
        ("Vector Size (Bytes)", size_label),
    ]

    p = figure(
        x_range=FactorRange(
            *implementations_sorted
        ),  # Implementations are the x-categories
        height=400,
        width=max(
            600, len(implementations_sorted) * 80
        ),  # Adjust width based on number of implementations
        title=f"Execution Time for {group_name} (Size: {size_label}, n: {size_elements})",
        toolbar_location=None,  # Can be "right" or None
        tools="hover,save",  # Simpler tools for individual bar charts
        tooltips=tooltips,
    )

    # Use a fixed palette or map implementations to colors if you want consistency across plots
    # For simplicity, let factor_cmap handle it for this single plot
    palette = (
        Bright6
        if len(implementations_sorted) <= 6
        else Category10[max(3, len(implementations_sorted))]
    )

    p.vbar(
        x="Implementation",
        top="Time_ms",
        source=source,
        width=0.7,
        # Color by implementation
        fill_color=factor_cmap(
            "Implementation", palette=palette, factors=implementations_sorted
        ),
        line_color="black",
    )

    p.xgrid.grid_line_color = None
    p.y_range.start = 0  # Time starts at 0
    p.xaxis.major_label_orientation = 0.8  # Radians, or "vertical"
    p.yaxis.axis_label = "Mean Time (milliseconds)"
    p.xaxis.axis_label = "Implementation"
    p.yaxis.formatter = NumeralTickFormatter(format="0,0.000a")

    show(p)
    print(f"Bokeh bar chart saved to {output_filename}")


def main():
    all_data = []

    for group_name in BENCHMARK_GROUPS:
        group_dir = os.path.join(CRITERION_BASE_DIR, group_name)
        if not os.path.isdir(group_dir):
            print(f"Benchmark group directory not found: {group_dir}")
            continue

        print(f"\nProcessing group: {group_name}")

        for impl_dir_name in os.listdir(group_dir):
            impl_path_base = os.path.join(group_dir, impl_dir_name)
            if not os.path.isdir(impl_path_base):
                continue

            for size_dir_name in os.listdir(impl_path_base):
                size_path = os.path.join(impl_path_base, size_dir_name)
                if not os.path.isdir(size_path):
                    continue

                estimates_file = os.path.join(size_path, "new", "estimates.json")
                if not os.path.exists(estimates_file):
                    estimates_file = os.path.join(size_path, "estimates.json")

                if os.path.exists(estimates_file):
                    mean_time_ns = parse_estimates(estimates_file)
                    if mean_time_ns is not None:
                        clean_impl_name = impl_dir_name.replace("_", " ")
                        clean_impl_name = re.sub(r"\s*\)\s*$", ")", clean_impl_name)
                        clean_impl_name = re.sub(r"\(\s+", "(", clean_impl_name)
                        clean_impl_name = re.sub(r"\s+\)", ")", clean_impl_name)

                        all_data.append(
                            {
                                "Group": group_name,
                                "Implementation": clean_impl_name.strip(),
                                "Size": int(size_dir_name),  # Number of elements
                                "Time (ns)": mean_time_ns,
                            }
                        )
                    else:
                        print(f"Could not parse mean time from {estimates_file}")
                else:
                    print(f"Estimates file not found: {estimates_file}")

    if not all_data:
        print("No benchmark data found. Exiting.")
        return

    df_all = pd.DataFrame(all_data)
    df_all["Time_ms"] = df_all["Time (ns)"] / 1_000_000
    df_all["Size_Bytes"] = df_all["Size"] * 4  # Assuming f32

    # Generate plots
    for group_name, df_group_data in df_all.groupby("Group"):
        # Plot line chart (performance over size for all implementations in the group)
        plot_line_chart_bokeh(group_name, df_group_data.copy())

        # Plot separate bar chart for each size in the group
        for size_elements, df_single_size_data in df_group_data.groupby("Size"):
            size_bytes_val = df_single_size_data["Size_Bytes"].iloc[
                0
            ]  # Get Size_Bytes for this group
            size_label_str = format_bytes_for_label(size_bytes_val)
            plot_grouped_bar_chart_per_size_bokeh(
                group_name, df_single_size_data.copy(), size_elements, size_label_str
            )


import numpy as np


def qr():
    # Define matrix A and vector b
    # A = np.array([[1.0, 1.0], [1.0, 2.0]])
    # b = np.array([3.0, 5.0])

    A = np.array([[1.0000, 0.0000], [0.0000, 1.0000], [1.0000, 1.0000]])
    b = np.array([1.0000, 2.0000, 4.0000])

    # QR decomposition
    Q, R = np.linalg.qr(A)

    # Compute Q^T * b
    Qt_b = np.dot(Q.T, b)

    # Solve R * x = Q^T * b using back substitution
    x = np.linalg.solve(R, Qt_b)

    print("Solution x:", x)
    print("A@x:", A @ x)


if __name__ == "__main__":
    qr()
    # main()
