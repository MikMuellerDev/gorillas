import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch

def visualize_embeddings(embeddings, labels=None, method='tsne', figsize=(10, 6), 
                        title='ReID Embeddings Visualization', save_path=None):
    """
    Visualize high-dimensional ReID embeddings in 2D space.
    
    Parameters:
    -----------
    embeddings : torch.Tensor or numpy.ndarray
        N x D array where N is number of samples and D is embedding dimension
    labels : array-like, optional
        Labels for each embedding (for coloring clusters)
    method : str, default='tsne'
        Dimensionality reduction method: 'tsne', 'pca', or 'both'
    figsize : tuple, default=(12, 8)
        Figure size
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    # Convert to numpy if tensor
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().detach().numpy()
    
    # Normalize embeddings (common practice for ReID)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    if method == 'both':
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*1.5, figsize[1]))
        
        # t-SNE
        print("Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        emb_2d_tsne = tsne.fit_transform(embeddings)
        
        # PCA
        print("Computing PCA...")
        pca = PCA(n_components=2, random_state=42)
        emb_2d_pca = pca.fit_transform(embeddings)
        
        # Plot t-SNE
        scatter1 = axes[0].scatter(emb_2d_tsne[:, 0], emb_2d_tsne[:, 1], 
                                   c=labels, cmap='hsv', alpha=0.6, s=20)
        axes[0].set_title(f't-SNE Projection\n{title}')
        axes[0].set_xlabel('Component 1')
        axes[0].set_ylabel('Component 2')
        
        # Plot PCA
        scatter2 = axes[1].scatter(emb_2d_pca[:, 0], emb_2d_pca[:, 1], 
                                   c=labels, cmap='hsv', alpha=0.6, s=20)
        axes[1].set_title(f'PCA Projection (Var: {pca.explained_variance_ratio_.sum():.2%})\n{title}')
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        
        if labels is not None:
            fig.colorbar(scatter1, ax=axes[0], label='Cluster ID')
            fig.colorbar(scatter2, ax=axes[1], label='Cluster ID')
            
    else:
        fig, ax = plt.subplots(figsize=figsize)
        
        if method == 'tsne':
            print("Computing t-SNE...")
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            emb_2d = reducer.fit_transform(embeddings)
            method_name = 't-SNE'
            var_info = ''
        else:  # pca
            print("Computing PCA...")
            reducer = PCA(n_components=2, random_state=42)
            emb_2d = reducer.fit_transform(embeddings)
            method_name = 'PCA'
            var_info = f' (Explained Var: {reducer.explained_variance_ratio_.sum():.2%})'
        
        scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], 
                           c=labels, cmap='hsv', alpha=0.6, s=20)
        ax.set_title(f'{method_name} Projection{var_info}\n{title}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        
        if labels is not None:
            plt.colorbar(scatter, ax=ax, label='Cluster ID')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=40, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    print(f"\nVisualization complete!")
    print(f"Original embedding dimension: {embeddings.shape[1]}")
    print(f"Number of samples: {embeddings.shape[0]}")
    if labels is not None:
        print(f"Number of unique clusters: {len(np.unique(labels))}")


# Example usage:
if __name__ == "__main__":
    # Generate example data (replace with your actual embeddings)
    n_samples = 500
    n_dims = 256
    n_clusters = 10
    
    # Simulate embeddings with cluster structure
    embeddings = []
    labels = []
    for i in range(n_clusters):
        cluster_center = np.random.randn(n_dims)
        cluster_samples = np.random.randn(n_samples // n_clusters, n_dims) * 0.3 + cluster_center
        embeddings.append(cluster_samples)
        labels.extend([i] * (n_samples // n_clusters))
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    # Visualize
    visualize_embeddings(embeddings, labels, method='both')