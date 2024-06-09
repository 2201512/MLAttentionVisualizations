from transformers import pipeline, GPT2Tokenizer, GPT2Model
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Step 1: Set up the pipeline
# This creates a text generation pipeline using the GPT-2 model. 
# The pipeline allows us to generate text based on a given input prompt.
pipe = pipeline("text-generation", model="openai-community/gpt2")

# Step 2: Generate text and retrieve attention weights
# Initialize the tokenizer and model for GPT-2. 
# Setting 'output_attentions=True' ensures that the model returns attention weights.
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2Model.from_pretrained("openai-community/gpt2", output_attentions=True)

# Define the input text
text = "A new sentence to visualize attention weights"

# Tokenize the input text to convert it into a format suitable for the model
inputs = tokenizer(text, return_tensors="pt")

# Pass the tokenized input through the model to get the output and attention weights
outputs = model(**inputs)

# Extract attention weights from the last layer (layer -1)
# Attention weights indicate how much focus the model places on different tokens when processing the input.
attention = outputs.attentions[-1]

# Aggregate attention heads by taking the mean
# This simplifies the visualization by averaging the attention scores from multiple heads.
attention = torch.mean(attention, dim=1).squeeze().detach().numpy()

# Define a function to plot the attention heatmap
# This function creates a heatmap of attention weights, showing the relationship between input tokens.
def plot_attention_heatmap(attention_weights, input_tokens, title="Attention Heatmap"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, xticklabels=input_tokens, yticklabels=input_tokens, cmap="viridis")
    plt.title(title)
    plt.xlabel("Input Tokens")
    plt.ylabel("Output Tokens")
    plt.show()

# Convert token IDs back to token strings for labeling the heatmap
input_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

# Plot the attention heatmap
plot_attention_heatmap(attention, input_tokens)