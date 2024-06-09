from transformers import GPT2Tokenizer, GPT2Model, utils
from bertviz import head_view, model_view
import torch

# Suppress standard warnings
utils.logging.set_verbosity_error()

# Load the tokenizer and model with output_attentions=True
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_attentions=True)

# Define the input text
text = "A new sentence to visualize attention weights"

# Tokenize the input text
inputs = tokenizer.encode(text, return_tensors='pt')

# Pass the inputs through the model to get the outputs, including attention weights
outputs = model(inputs)

# Extract attention weights from the outputs
attention = outputs.attentions

# Convert token IDs to tokens
tokens = tokenizer.convert_ids_to_tokens(inputs[0])

# Generate the head view HTML representation
html_head_view = head_view(attention, tokens, html_action='return')

# Save the head view HTML to a file
with open("head_view.html", 'w') as file:
    file.write(html_head_view.data)

# Generate the model view HTML representation
html_model_view = model_view(attention, tokens, html_action='return')

# Save the model view HTML to a file
with open("model_view.html", 'w') as file:
    file.write(html_model_view.data)

# Optionally, display the HTML in Jupyter (if applicable)
import IPython.display as display
display.display(html_head_view)
display.display(html_model_view)
