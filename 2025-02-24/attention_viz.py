import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def visualize_attention(sentence, attention_weights):
    """
    Visualize attention weights between words in a sentence

    Args:
        sentence (list): List of tokens/words
        attention_weights (numpy.ndarray): 2D array with attention weights
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create heatmap
    sns.heatmap(
        attention_weights,
        xticklabels=sentence,
        yticklabels=sentence,
        cmap="YlGnBu",
        annot=True,
        cbar=True,
        square=True,
        ax=ax
    )

    plt.title("Attention Weights Visualization")
    plt.tight_layout()
    plt.savefig("attention_visualization.png")
    plt.show()


# Example usage
if __name__ == "__main__":
    # Sample sentence
    sentence = ["I", "love", "studying", "machine", "learning", "models"]

    # Create fake attention weights (normally this would come from a model)
    # In a real transformer, this would be one of the attention heads
    n = len(sentence)
    attention = np.zeros((n, n))

    # Create some meaningful pattern for demonstration
    for i in range(n):
        for j in range(n):
            # Make words attend to themselves and related words more
            if i == j:  # Self-attention is strong
                attention[i, j] = 0.5
            elif abs(i - j) == 1:  # Adjacent words have moderate attention
                attention[i, j] = 0.2
            else:  # Other words have small attention
                attention[i, j] = 0.1 / (abs(i - j))

    # Normalize rows to sum to 1 (like softmax would do)
    attention = attention / attention.sum(axis=1, keepdims=True)

    # Visualize the attention weights
    visualize_attention(sentence, attention)

    print("Attention visualization saved as 'attention_visualization.png'")