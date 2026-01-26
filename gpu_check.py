import torch
import ollama

print(f"Is CUDA (GPU) available for Torch? {torch.cuda.is_available()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")

try:
    models = ollama.list()
    print(models)
    print("Ollama App is running and connected!")
except:
    print("Ollama App is NOT running. Please open the Ollama App from your taskbar.")