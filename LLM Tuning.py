import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel

# Function to load the model and tokenizer
def load_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    return tokenizer, model, device

# Function to generate responses for multiple inputs in a batched manner
def generate_responses(inputs, max_length=256, do_sample=True, top_k=50, top_p=0.95, temperature=0.7):
    # Tokenize the input texts
    inputs = [tokenizer.encode(input_text, return_tensors="pt").to(device) for input_text in inputs]
    
    # Combine all inputs into a single batch
    inputs = torch.cat(inputs, dim=0)
    
    # Generate responses using the model with mixed precision and parallel processing
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    
    with autocast():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature
        )
    
    # Decode the generated tokens back to text
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    return responses

# Additional optimizations for better performance
def optimize_model(model):
    if torch.cuda.is_available():
        print("Using GPU")
        model.to(device)
    else:
        print("Using CPU")

    # Enable mixed precision training to reduce memory usage and improve speed
    scaler = GradScaler()

    # Optimize the model for inference using TorchScript or ONNX
    scripted_model = torch.jit.script(model)

    # Use DataParallel for multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    
    return scripted_model, scaler

# Example usage
model_name = "Qwen-2.5-75B"
tokenizer, model, device = load_model(model_name)
scripted_model, scaler = optimize_model(model)

# Example of using the optimized model with batch processing
input_texts = [
    "What is the capital of France?",
    "Explain the concept of artificial intelligence.",
    "Describe the process of photosynthesis."
]

responses = generate_responses(input_texts, max_length=512)
for i, response in enumerate(responses):
    print(f"Input {i+1}: {input_texts[i]}")
    print(f"Response: {response}\n")
