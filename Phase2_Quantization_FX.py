import torch
import torch.quantization
from torch.ao.quantization import quantize_fx
import copy
import time
import os
import warnings
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import matplotlib.pyplot as plt
import importlib

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from Phase 1 setup
from baseline_setup import get_dataloaders, get_model, split_and_verify_model, test_accuracy
from attacker import Decoder, ATTACKER_MODEL_PATH

# --- Configuration ---
BACKEND_ENGINE = 'qnnpack'
MODEL_PATH_INT8_EDGE = "models/edge_model_int8_fx.pth"
CPU_DEVICE = torch.device("cpu")

# ==========================================
# 1. QUANTIZATION (FX GRAPH MODE)
# ==========================================
def quantize_edge_model_fx(edge_model_fp32, trainloader):
    print("\n--- Starting FX Graph Mode Quantization ---")

    edge_model_fp32.eval().to(CPU_DEVICE)
    example_input = torch.randn(1, 3, 224, 224).to(CPU_DEVICE)

    qconfig_dict = {"": torch.quantization.get_default_qconfig(BACKEND_ENGINE)}

    print("Preparing model (tracing and fusing)...")
    model_prepared = quantize_fx.prepare_fx(edge_model_fp32, qconfig_dict, example_input)

    print("Calibrating model...")
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(trainloader)):
            if i >= 20: break
            model_prepared(images.to(CPU_DEVICE))

    print("Converting to INT8...")
    model_int8 = quantize_fx.convert_fx(model_prepared)

    torch.save(model_int8.state_dict(), MODEL_PATH_INT8_EDGE)
    return model_int8

# ==========================================
# 2. EVALUATION FUNCTIONS
# ==========================================
def evaluate_pipeline_accuracy(edge_model, server_model, testloader, name="Model"):
    print(f"\n--- Evaluating Accuracy: {name} ---")
    edge_model.eval().to(CPU_DEVICE)
    server_model.eval().to(CPU_DEVICE)

    correct = 0
    total = 0

    # Limit evaluation to 100 batches to save time
    LIMIT_BATCHES = 100

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(testloader, desc="Testing Accuracy")):
            if i >= LIMIT_BATCHES: break

            images, labels = images.to(CPU_DEVICE), labels.to(CPU_DEVICE)

            # Pass through Edge
            features = edge_model(images)
            # Pass through Server
            outputs = server_model(features)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"{name} Accuracy: {acc:.2f}%")
    return acc

def evaluate_efficiency(fp32_model, int8_model):
    print("\n--- Efficiency Evaluation (CPU) ---")

    # Size
    torch.save(fp32_model.state_dict(), "temp_fp32.pth")
    torch.save(int8_model.state_dict(), "temp_int8.pth")
    size_fp32 = os.path.getsize("temp_fp32.pth") / (1024 * 1024)
    size_int8 = os.path.getsize("temp_int8.pth") / (1024 * 1024)
    print(f"FP32 Size: {size_fp32:.2f} MB")
    print(f"INT8 Size: {size_int8:.2f} MB")
    print(f"Reduction: {size_fp32 / size_int8:.1f}x")

    # Latency
    dummy = torch.randn(1, 3, 224, 224).to(CPU_DEVICE)
    fp32_model.to(CPU_DEVICE)

    # Warmup
    for _ in range(5): _ = fp32_model(dummy)

    # Measure FP32
    start = time.perf_counter()
    for _ in range(20): _ = fp32_model(dummy)
    lat_fp32 = (time.perf_counter() - start) / 20 * 1000

    # Measure INT8
    for _ in range(5): _ = int8_model(dummy)
    start = time.perf_counter()
    for _ in range(20): _ = int8_model(dummy)
    lat_int8 = (time.perf_counter() - start) / 20 * 1000

    print(f"FP32 Latency: {lat_fp32:.2f} ms")
    print(f"INT8 Latency: {lat_int8:.2f} ms")
    print(f"Speedup: {lat_fp32 / lat_int8:.2f}x")

    return size_fp32, size_int8, lat_fp32, lat_int8

def evaluate_privacy_attack_local(edge_model, decoder, testloader):
    # Local version of evaluate_attack to ensure everything stays on CPU
    edge_model.eval().to(CPU_DEVICE)
    decoder.eval().to(CPU_DEVICE)

    total_psnr = 0
    batches = 0
    limit = 20 # limit batches to save time

    with torch.no_grad():
        for i, (images, _) in enumerate(testloader):
            if i >= limit: break
            images = images.to(CPU_DEVICE)

            features = edge_model(images)
            recons = decoder(features)

            for j in range(len(images)):
                img_orig = (images[j] * 0.5 + 0.5).permute(1, 2, 0).numpy().clip(0, 1)
                img_recon = (recons[j] * 0.5 + 0.5).permute(1, 2, 0).numpy().clip(0, 1)
                total_psnr += psnr(img_orig, img_recon, data_range=1.0)
            batches += len(images)

    return total_psnr / batches

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    trainloader, testloader = get_dataloaders()

    # Load Models
    print("Loading Models...")
    full_model = get_model().to(CPU_DEVICE)
    full_model.load_state_dict(torch.load("models/mobilenet_v3_full_fp32.pth", map_location=CPU_DEVICE))

    edge_fp32, server_fp32 = split_and_verify_model(full_model)
    edge_fp32.to(CPU_DEVICE)
    server_fp32.to(CPU_DEVICE)

    # Quantize
    edge_int8 = quantize_edge_model_fx(edge_fp32, trainloader)

    # --- NEW: ACCURACY CHECK ---
    print("\n--- Utility (Accuracy) Evaluation ---")
    acc_fp32 = evaluate_pipeline_accuracy(edge_fp32, server_fp32, testloader, "FP32 Pipeline")
    acc_int8 = evaluate_pipeline_accuracy(edge_int8, server_fp32, testloader, "INT8 Pipeline")
    print(f"\nAccuracy Drop: {acc_fp32 - acc_int8:.2f}%")
    if (acc_fp32 - acc_int8) < 5.0:
        print("SUCCESS: Accuracy preserved within 5% margin.")
    else:
        print("WARNING: Significant accuracy drop observed.")

    # Evaluate Efficiency
    evaluate_efficiency(edge_fp32, edge_int8)

    # Evaluate Privacy
    print("\n--- Privacy Evaluation ---")
    decoder = Decoder().to(CPU_DEVICE)
    decoder.load_state_dict(torch.load("models/decoder_attacker.pth", map_location=CPU_DEVICE))

    psnr_fp32 = evaluate_privacy_attack_local(edge_fp32, decoder, testloader)
    print(f"Attacker Quality (FP32): {psnr_fp32:.2f} dB")

    psnr_int8 = evaluate_privacy_attack_local(edge_int8, decoder, testloader)
    print(f"Attacker Quality (INT8): {psnr_int8:.2f} dB")

    print(f"\nPrivacy Gain: {psnr_fp32 - psnr_int8:.2f} dB (Higher is Better)")

    # Visual Check
    with torch.no_grad():
        inputs, _ = next(iter(testloader))
        inputs = inputs.to(CPU_DEVICE)
        feat_int8 = edge_int8(inputs)
        recon_int8 = decoder(feat_int8)

        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1); plt.imshow((inputs[0]*0.5+0.5).permute(1,2,0).numpy().clip(0,1)); plt.title("Original")
        plt.subplot(1,2,2); plt.imshow((recon_int8[0]*0.5+0.5).permute(1,2,0).numpy().clip(0,1)); plt.title("INT8 Recon")
        plt.savefig('debug_int8_check.png')

if __name__ == "__main__":
    main()
