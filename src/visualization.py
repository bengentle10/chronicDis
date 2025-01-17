import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Set Seaborn style for all plots
sns.set_theme(style="whitegrid")  # This replaces both sns.set_style and plt.style.use
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def plot_disease_distribution(disease_counts):
    """Plot distribution of child diseases."""
    plt.figure(figsize=(12, 6))
    sns.histplot(data=disease_counts['child_count'], bins=50)
    plt.title('Distribution of Child Diseases')
    plt.xlabel('Number of Child Diseases')
    plt.ylabel('Count')
    plt.savefig('disease_distribution.png')
    plt.close()

def plot_disease_hierarchy(df):
    """Plot distribution of leaf vs non-leaf diseases."""
    plt.figure(figsize=(10, 6))
    leaf_counts = df['ontology'].apply(lambda x: x.get('leaf', False)).value_counts()
    leaf_counts.plot(kind='bar')
    plt.title('Distribution of Leaf vs Non-leaf Diseases')
    plt.xlabel('Is Leaf Disease')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('hierarchy_distribution.png')
    plt.close()

def visualize_disease_network(G):
    """Enhanced network visualization."""
    plt.figure(figsize=(20, 20))
    
    # Use spring layout with adjusted parameters
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Calculate node sizes based on degree centrality
    centrality = nx.degree_centrality(G)
    
    # Create a custom colormap for nodes
    main_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'main']
    child_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'child']
    
    # Draw main nodes
    if main_nodes:
        main_sizes = [5000 for _ in main_nodes]  # Fixed size for main nodes
        nx.draw_networkx_nodes(G, pos, 
                             nodelist=main_nodes,
                             node_color='#FF9999',
                             node_size=main_sizes,
                             alpha=0.8)
    
    # Draw child nodes
    if child_nodes:
        child_sizes = [3000 for _ in child_nodes]  # Fixed size for child nodes
        nx.draw_networkx_nodes(G, pos,
                             nodelist=child_nodes,
                             node_color='#99FF99',
                             node_size=child_sizes,
                             alpha=0.6)
    
    # Draw edges with varying width based on weight
    edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos,
                          width=[w*2 for w in edge_weights],
                          edge_color='gray',
                          alpha=0.5,
                          arrows=True,
                          arrowsize=20)
    
    # Add labels with custom formatting
    labels = nx.get_node_attributes(G, 'name')
    nx.draw_networkx_labels(G, pos,
                           labels,
                           font_size=8,
                           font_weight='bold',
                           font_color='black')
    
    plt.title('Disease Hierarchy Network\nNode size indicates node type',
              fontsize=16, pad=20)
    plt.axis('off')
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor='#FF9999', markersize=15,
                                 label='Main Disease'),
                      plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor='#99FF99', markersize=10,
                                 label='Child Disease')]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=12)
    
    plt.savefig('network_visualization.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.close()

def plot_training_history(history):
    """Plot training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Training Loss', color='blue', alpha=0.7)
    ax1.plot(history['val_loss'], label='Validation Loss', color='red', alpha=0.7)
    ax1.set_title('Model Loss Over Time', fontsize=12)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Training Accuracy', color='blue', alpha=0.7)
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='red', alpha=0.7)
    ax2.set_title('Model Accuracy Over Time', fontsize=12)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_analysis(y_true, y_pred, y_prob):
    """Create comprehensive prediction analysis visualizations."""
    fig = plt.figure(figsize=(20, 10))
    
    # Confusion Matrix
    ax1 = fig.add_subplot(231)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # ROC Curve
    ax2 = fig.add_subplot(232)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_title('ROC Curve')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend()
    
    # Prediction Distribution
    ax3 = fig.add_subplot(233)
    # Create DataFrame for histogram
    plot_data = pd.DataFrame({
        'Probability': y_prob.flatten(),
        'True_Label': ['Positive' if y == 1 else 'Negative' for y in y_true]
    })
    
    sns.histplot(
        data=plot_data,
        x='Probability',
        hue='True_Label',
        bins=50,
        ax=ax3,
        palette={'Positive': 'blue', 'Negative': 'red'},
        alpha=0.5
    )
    
    ax3.set_title('Prediction Probability Distribution by Class')
    ax3.set_xlabel('Predicted Probability')
    ax3.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
