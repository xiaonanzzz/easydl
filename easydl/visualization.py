import pandas as pd
import plotly.graph_objects as go
from typing import Optional
import plotly.express as px

def plot_precision_recall_vs_threshold_curve(precision_list: list, recall_list: list, threshold_list: list):
    """
    Visualize the Precision-Recall Curve.
    """
    print(len(threshold_list), len(precision_list), len(recall_list))
    df = pd.DataFrame({'threshold': threshold_list, 'precision': precision_list[:-1], 'recall': recall_list[:-1]})
    fig = px.line(df, x='threshold', y=['precision', 'recall'])
    
    return fig

def plot_precision_vs_recall_curve(precision_list: list, recall_list: list):
    """
    Visualize the Precision-Recall Curve.
    """
    df = pd.DataFrame({'precision': precision_list, 'recall': recall_list})
    fig = px.line(df, x='recall', y='precision')
    return fig