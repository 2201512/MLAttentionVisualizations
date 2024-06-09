Simple HeatMap Using GPT-2 
```markdown
# Visualization Tools for Self-Attention Mechanisms

## Introduction
This document categorizes various visualization tools for self-attention mechanisms.

*Coloured text are Hyperlinked*

## Simple Heatmap using MatplotLib for NLP
Use GPT-2 as the model and ascertain the attention weight in the last layer, for the sentence "A new sentence to visualize attention weights".

```python
from transformers import pipeline, GPT2Tokenizer, GPT2Model
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Step 1: Set up the pipeline
pipe = pipeline("text-generation", model="openai-community/gpt2")

# Step 2: Generate text and retrieve attention weights
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2Model.from_pretrained("openai-community/gpt2", output_attentions=True)

text = "A new sentence to visualize attention weights"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

attention = outputs.attentions[-1]
attention = torch.mean(attention, dim=1).squeeze().detach().numpy()

def plot_attention_heatmap(attention_weights, input_tokens, title="Attention Heatmap"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, xticklabels=input_tokens, yticklabels=input_tokens, cmap="viridis")
    plt.title(title)
    plt.xlabel("Input Tokens")
    plt.ylabel("Output Tokens")
    plt.show()

input_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
plot_attention_heatmap(attention, input_tokens)
```

## BertViz
[BertViz](https://github.com/jessevig/bertviz)
- Allows for head, model, and neuron view
- Seems more applicable to NLP


```python
from transformers import GPT2Tokenizer, GPT2Model, utils
from bertviz import head_view, model_view
import torch

# Suppress standard warnings
utils.logging.set_verbosity_error()

# Load the tokenizer and model with output_attentions=True
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_attentions=True)

text = "A new sentence to visualize attention weights"
inputs = tokenizer.encode(text, return_tensors='pt')
outputs = model(inputs)

attention = outputs.attentions
tokens = tokenizer.convert_ids_to_tokens(inputs[0])

html_head_view = head_view(attention, tokens, html_action='return')
with open("head_view.html", 'w') as file:
    file.write(html_head_view.data)

html_model_view = model_view(attention, tokens, html_action='return')
with open("model_view.html", 'w') as file:
    file.write(html_model_view.data)

import IPython.display as display
display.display(html_head_view)
display.display(html_model_view)
```

## AttentionViz
[AttentionViz](https://github.com/catherinesyeh/attention-viz?tab=readme-ov-file)
- Matrix view to view all attention heads
- Single view to seek each attention head
- Image/sentence view for patterns
- Applicable to both visual and NLP

## Attention by Matt Neary
[Attention by Matt Neary](https://github.com/mattneary/attention?tab=readme-ov-file)
- Visualization using normalized means (sigmoid function)
- Applicable to NLP

## More Attention Visualization Libraries Using Heatmaps
- [Attention Transfer](https://github.com/szagoruyko/attention-transfer/tree/master)
- [Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability?tab=readme-ov-file)

## Visualization Libraries for Saliency
- [Learning Interpretability Tool (LIT)](https://github.com/PAIR-code/lit)
- [Ecco](https://github.com/jalammar/ecco?tab=readme-ov-file)
- [Transformers-Interpret](https://github.com/cdpierse/transformers-interpret?tab=readme-ov-file)

## General Purpose Visualization Libraries
- [TensorBoard for TensorFlow](https://github.com/tensorflow/tensorboard)
- [Captum for PyTorch](https://github.com/pytorch/captum)
```
