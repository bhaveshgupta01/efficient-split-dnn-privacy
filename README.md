# Efficient and Private: Analyzing Quantization on Split-DNN Security

**NYU Efficient AI - Final Project**

This project investigates the critical intersection of **model efficiency** and **data privacy** in Split Deep Neural Network (DNN) systems designed for edge devices.

## 🎯 Project Goal

In many real-world "edge" applications (like AR glasses or smart cameras), a **Split-DNN** architecture is used. The on-device "edge" model runs the first few layers, and the resulting "features" are sent to a server for final processing.

This creates two major problems:
1.  **Efficiency:** The on-device model must be extremely small and fast.
2.  **Privacy:** The intermediate features can be intercepted and "inverted" by an attacker to reconstruct the original, private input (e.g., a user's face).

**This project explores a key question: Can we solve both problems at once?**

My hypothesis is that **INT8 quantization** (a popular efficiency technique) can serve a dual purpose:
1.  **Efficiency:** Reduces memory usage and latency on the edge device.
2.  **Privacy Defense:** Acts as a form of "lossy compression," introducing quantization noise that makes feature inversion significantly harder for an attacker.

---

## 📊 Key Results

We compared the Baseline (FP32) Split-DNN against the Quantized (INT8) Split-DNN. The results confirm that quantization significantly degrades the attacker's ability to reconstruct the original image while maintaining model accuracy.

### 1. Privacy Visualized
The image below shows the attacker's reconstruction attempt.
* **Top Row:** Original Input Images (CIFAR-100).
* **Middle Row:** Reconstruction from **FP32** features (High Privacy Risk).
* **Bottom Row:** Reconstruction from **INT8** features (Improved Privacy).

![Privacy Comparison](final_privacy_comparison.png)

### 2. Performance Trade-offs
*Summary of metrics observed during the experiment:*

| Metric | Baseline (FP32) | Quantized (INT8) | Improvement |
| :--- | :--- | :--- | :--- |
| **Model Size (Edge)** | ~6.0 MB | ~1.6 MB | **4x Smaller** |
| **Privacy (PSNR)** | High (Easier to reconstruct) | Low (Harder to reconstruct) | **Better Privacy** |
| **Accuracy** | Baseline | < 1% Drop | **Maintained** |

*(Note: Lower PSNR on the reconstruction attack indicates BETTER privacy, as the reconstructed image is further from the original.)*

---

## 🛠 Methodology

This project is implemented in three phases:

1.  **Baseline System (FP32):**
    * Fine-tune a **MobileNetV3-Small** model on the **CIFAR-100** dataset.
    * Split the model into an `edge_model` (first few layers) and `server_model`.
    * Train a `Decoder` (Attacker) to invert the FP32 features back into images.

2.  **Quantized System (INT8):**
    * Apply **PyTorch FX Graph Mode Quantization** to the `edge_model`.
    * Calibrate the model using a representative dataset to ensure accuracy.
    * Convert the edge layers to run using INT8 integer arithmetic.

3.  **Analysis & Comparison:**
    * Run the *same* `Decoder` attack on the new INT8 features.
    * Measure the drop in reconstruction quality (PSNR) and the reduction in model size.

---

## 💻 Tech Stack

* **Framework:** PyTorch & PyTorch FX
* **Architecture:** MobileNetV3-Small (Split-DNN)
* **Dataset:** CIFAR-100
* **Quantization:** `torch.quantization.quantize_fx` (Post-Training Quantization)
* **Metrics:** `skimage.metrics.peak_signal_noise_ratio` (PSNR)

---

## 📂 File Structure

* `baseline_setup.py`: Downloads data, fine-tunes the base MobileNet model, and saves the weights.
* `attacker.py`: Trains the Decoder network to reconstruct images from intermediate feature maps.
* `Phase2_Quantization_FX.py`: Performs the INT8 quantization, evaluates accuracy, and runs the privacy benchmark (generating the comparison plots).
* `final_privacy_comparison.png`: The visual output of the experiment.

---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/bhaveshgupta01/efficient-split-dnn-privacy.git](https://github.com/bhaveshgupta01/efficient-split-dnn-privacy.git)
    cd efficient-split-dnn-privacy
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch torchvision scikit-image matplotlib tqdm
    ```

3.  **Run the Baseline Setup:**
    (This trains the model and the attacker on FP32 data)
    ```bash
    python baseline_setup.py
    python attacker.py
    ```

4.  **Run Quantization & Evaluation:**
    (This quantizes the model to INT8, runs the attack, and generates the comparison graph)
    ```bash
    python Phase2_Quantization_FX.py
    ```

---

## ✅ Project Roadmap

* [x] **Phase 1: Baseline System**
    * [x] Fine-tune MobileNetV3-Small on CIFAR-100.
    * [x] Implement Split-DNN architecture.
    * [x] Train Decoder (Attacker) and measure baseline privacy risks.
* [x] **Phase 2: Quantization**
    * [x] Implement Post-Training Quantization using PyTorch FX.
    * [x] Calibrate and convert the `edge_model` to INT8.
* [x] **Phase 3: Evaluation**
    * [x] Evaluate Accuracy drop (FP32 vs INT8).
    * [x] Evaluate Model Size reduction (4x compression).
    * [x] visual comparison of privacy leakage.
