import numpy as np
import matplotlib.pyplot as plt
import torch


def get_positional_encoding(seq_len, d_model):
    """
    Compute positional encoding as described in 'Attention Is All You Need'

    Args:
        seq_len (int): Maximum sequence length
        d_model (int): Dimension of the model

    Returns:
        numpy.ndarray: Positional encoding matrix of shape (seq_len, d_model)
    """
    # Initialize positional encoding matrix
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    # Create empty matrix
    pe = np.zeros((seq_len, d_model))

    # Apply sine to even indices
    pe[:, 0::2] = np.sin(position * div_term)

    # Apply cosine to odd indices
    pe[:, 1::2] = np.cos(position * div_term)

    return pe


def visualize_positional_encoding(pe_matrix, save_path=None):
    """
    Visualize positional encoding matrix as a heatmap

    Args:
        pe_matrix (numpy.ndarray): Positional encoding matrix
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(pe_matrix, cmap='viridis', aspect='auto')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Sequence Position')
    plt.title('Positional Encoding Visualization')
    plt.colorbar()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def visualize_pe_patterns(pe_matrix, positions=[0, 10, 20, 30], save_path=None):
    """
    Visualize patterns in the positional encoding for specific positions

    Args:
        pe_matrix (numpy.ndarray): Positional encoding matrix
        positions (list): List of positions to visualize
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(12, 6))

    for pos in positions:
        if pos < pe_matrix.shape[0]:
            plt.plot(pe_matrix[pos, :], label=f'Position {pos}')

    plt.xlabel('Embedding Dimension')
    plt.ylabel('Encoding Value')
    plt.title('Positional Encoding Patterns at Different Positions')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    # Parameters
    sequence_length = 100
    model_dimension = 128

    # Generate positional encoding
    pe = get_positional_encoding(sequence_length, model_dimension)

    # Visualize as heatmap
    visualize_positional_encoding(pe, "positional_encoding_heatmap.png")

    # Visualize patterns for different positions
    visualize_pe_patterns(pe, positions=[0, 25, 50, 75], save_path="positional_encoding_patterns.png")

    print("Positional encoding visualizations saved as PNG files.")