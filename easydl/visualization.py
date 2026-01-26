from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from seaborn import histplot
from sklearn.metrics import auc, precision_recall_curve

from easydl.numpyext import get_upper_triangle_values


def plot_precision_recall_vs_threshold_curve(
    precision_list: list, recall_list: list, threshold_list: list
):
    """
    Visualize the Precision-Recall Curve.
    """
    print(len(threshold_list), len(precision_list), len(recall_list))
    df = pd.DataFrame(
        {
            "threshold": threshold_list,
            "precision": precision_list[:-1],
            "recall": recall_list[:-1],
        }
    )
    fig = px.line(df, x="threshold", y=["precision", "recall"])

    return fig


def plot_precision_vs_recall_curve(precision_list: list, recall_list: list):
    """
    Visualize the Precision-Recall Curve.
    """
    df = pd.DataFrame({"precision": precision_list, "recall": recall_list})
    fig = px.line(df, x="recall", y="precision")
    return fig


class PairwiseScoreAnalysisPlot:
    """
    Class for analyzing pairwise score matrices with ground truth labels.
    """

    def __init__(self, pair_gt_matrix: np.ndarray, pair_score_matrix: np.ndarray):
        """
        Initialize with pairwise ground truth and score matrices.

        Args:
            pair_gt_matrix: 2D numpy array of ground truth pairwise labels (0 or 1)
            pair_score_matrix: 2D numpy array of pairwise similarity scores
        """
        self.pair_gt_matrix = pair_gt_matrix
        self.pair_score_matrix = pair_score_matrix

    def analyze(self, n_bins: int = 50):
        """
        Analyze the pairwise scores by extracting upper triangle values, creating bins,
        computing histograms per bin grouped by ground truth, and subsampling precision/recall
        to match bins. Results are stored in self.

        Args:
            n_bins: Number of bins for distribution analysis (default: 50)
        """
        # Generate upper triangle indices
        self.upper_triangle_index = np.triu_indices_from(self.pair_gt_matrix, k=1)

        # Extract upper triangle values
        self.pairwise_gt_list = self.pair_gt_matrix[self.upper_triangle_index]
        self.pairwise_score_list = self.pair_score_matrix[self.upper_triangle_index]

        # Create distribution bins
        self.distribution_bins = np.linspace(
            np.min(self.pairwise_score_list),
            np.max(self.pairwise_score_list),
            n_bins + 1,
        )

        # Calculate precision, recall, and thresholds
        precision_list_full, recall_list_full, threshold_list = precision_recall_curve(
            self.pairwise_gt_list, self.pairwise_score_list
        )

        # Store full precision and recall lists (with extra element)
        self.precision_list_full = precision_list_full
        self.recall_list_full = recall_list_full
        self.threshold_list = threshold_list

        # Remove the last element from precision and recall (they have one extra element)
        precision_list = precision_list_full[:-1]
        recall_list = recall_list_full[:-1]

        # Calculate histograms for each bin, grouped by ground truth (0 and 1)
        # First calculate raw counts
        raw_counts = {0: [], 1: []}

        for i in range(len(self.distribution_bins) - 1):
            bin_left = self.distribution_bins[i]
            bin_right = self.distribution_bins[i + 1]

            # Find scores in this bin
            mask = (self.pairwise_score_list >= bin_left) & (
                self.pairwise_score_list < bin_right
            )
            if i == len(self.distribution_bins) - 2:  # Last bin includes right edge
                mask = (self.pairwise_score_list >= bin_left) & (
                    self.pairwise_score_list <= bin_right
                )

            # Get scores and corresponding ground truth for this bin
            bin_scores = self.pairwise_score_list[mask]
            bin_gt = self.pairwise_gt_list[mask]

            # Calculate histogram counts for gt=0 and gt=1
            count_0 = np.sum(bin_gt == 0)
            count_1 = np.sum(bin_gt == 1)

            raw_counts[0].append(count_0)
            raw_counts[1].append(count_1)

        # Calculate bin width for density normalization
        bin_width = self.distribution_bins[1] - self.distribution_bins[0]

        # Normalize to density for each group separately
        # Density = count / (total_count * bin_width) so that integral = 1
        self.bin_histograms = {0: [], 1: []}

        # Get total counts for each group
        total_count_0 = np.sum(self.pairwise_gt_list == 0)
        total_count_1 = np.sum(self.pairwise_gt_list == 1)

        # Normalize each group separately
        for i in range(len(raw_counts[0])):
            if total_count_0 > 0:
                density_0 = raw_counts[0][i] / (total_count_0 * bin_width)
            else:
                density_0 = 0.0
            self.bin_histograms[0].append(density_0)

            if total_count_1 > 0:
                density_1 = raw_counts[1][i] / (total_count_1 * bin_width)
            else:
                density_1 = 0.0
            self.bin_histograms[1].append(density_1)

        # Subsample precision and recall to match bins
        # We'll map each threshold to a bin and aggregate precision/recall per bin
        bin_centers = (self.distribution_bins[:-1] + self.distribution_bins[1:]) / 2

        subsampled_precision = []
        subsampled_recall = []
        subsampled_thresholds = []

        for i in range(len(bin_centers)):
            bin_center = bin_centers[i]
            bin_left = self.distribution_bins[i]
            bin_right = self.distribution_bins[i + 1]

            # Find thresholds that fall within this bin
            if i == len(bin_centers) - 1:  # Last bin
                mask = (threshold_list >= bin_left) & (threshold_list <= bin_right)
            else:
                mask = (threshold_list >= bin_left) & (threshold_list < bin_right)

            if np.any(mask):
                # Average precision and recall for thresholds in this bin
                subsampled_precision.append(np.mean(precision_list[mask]))
                subsampled_recall.append(np.mean(recall_list[mask]))
                subsampled_thresholds.append(bin_center)
            else:
                # If no threshold in this bin, use nearest threshold
                if len(threshold_list) > 0:
                    nearest_idx = np.argmin(np.abs(threshold_list - bin_center))
                    subsampled_precision.append(precision_list[nearest_idx])
                    subsampled_recall.append(recall_list[nearest_idx])
                    subsampled_thresholds.append(bin_center)
                else:
                    subsampled_precision.append(0.0)
                    subsampled_recall.append(0.0)
                    subsampled_thresholds.append(bin_center)

        # Store subsampled results
        self.subsampled_precision = np.array(subsampled_precision)
        self.subsampled_recall = np.array(subsampled_recall)
        self.subsampled_thresholds = np.array(subsampled_thresholds)
        self.bin_centers = bin_centers

    def plot(self, figsize=(12, 6)):
        """
        Plot distribution of positive and negative values, and precision/recall against bins/thresholds.

        Args:
            figsize: Tuple of (width, height) for the figure size (default: (12, 6))

        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if not hasattr(self, "pairwise_gt_list"):
            raise ValueError("Must call analyze() before plot()")

        fig, ax1 = plt.subplots(figsize=figsize)

        # Use stored bin centers
        bin_width = self.distribution_bins[1] - self.distribution_bins[0]

        # Plot normalized density histogram for negative values (gt=0)
        ax1.bar(
            self.bin_centers,
            self.bin_histograms[0],
            width=bin_width * 0.8,
            alpha=0.6,
            label="Negative pairs (GT=0)",
            color="blue",
            align="center",
        )

        # Plot normalized density histogram for positive values (gt=1)
        ax1.bar(
            self.bin_centers,
            self.bin_histograms[1],
            width=bin_width * 0.8,
            alpha=0.6,
            label="Positive pairs (GT=1)",
            color="orange",
            align="center",
        )

        ax1.set_xlabel("Pairwise Score", fontsize=12)
        ax1.set_ylabel("Density", fontsize=12, color="black")
        ax1.tick_params(axis="y", labelcolor="black")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Create secondary y-axis for precision and recall
        ax2 = ax1.twinx()

        # Plot precision and recall against subsampled thresholds (bin centers)
        ax2.plot(
            self.subsampled_thresholds,
            self.subsampled_precision,
            color="green",
            label="Precision",
            linewidth=2,
            alpha=0.8,
            marker="o",
            markersize=3,
        )
        ax2.plot(
            self.subsampled_thresholds,
            self.subsampled_recall,
            color="red",
            label="Recall",
            linewidth=2,
            alpha=0.8,
            linestyle="--",
            marker="s",
            markersize=3,
        )

        ax2.set_ylabel("Precision / Recall", fontsize=12, color="black")
        ax2.tick_params(axis="y", labelcolor="black")
        ax2.set_ylim([0, 1])
        ax2.legend(loc="upper right")

        # Calculate PR AUC for title
        pr_auc = auc(self.recall_list_full, self.precision_list_full)

        total_pairs = len(self.pairwise_gt_list)
        positive_pairs = np.sum(self.pairwise_gt_list == 1)

        title = (
            f"Pairwise Score Distribution and Precision/Recall\n"
            f"Total pairs: {total_pairs}, Positive pairs: {positive_pairs}, PR AUC: {round(pr_auc, 4)}"
        )
        ax1.set_title(title, fontsize=14)

        plt.tight_layout()

        return fig
