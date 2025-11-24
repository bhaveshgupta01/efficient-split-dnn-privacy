# Efficient and Private: Analyzing Quantization on Split-DNN Security

This is the repository for my final project for the NYU Efficient AI course. This project investigates the critical intersection of model efficiency and data privacy in split Deep Neural Network (DNN) systems designed for edge devices.

### Project Goal

In many real-world "edge" applications (like AR glasses or smart cameras), a **Split-DNN** architecture is used. The on-device "edge" model runs the first few layers, and the resulting "features" are sent to a server for final processing.

This creates two major problems:

1. **Efficiency:** The on-device model must be extremely small and fast.

2. **Privacy:** The intermediate features can be intercepted and "inverted" by an attacker to reconstruct the original, private input (e.g., a user's face).

**This project explores a key question: Can we solve both problems at once?**

My hypothesis is that **INT8 quantization** (a popular efficiency technique) can serve a dual purpose:

1. It will make the edge model faster and smaller.

2. It will act as a form of "noise" or "lossy compression," making the features harder for an attacker to reconstruct, thus improving privacy.

### Methodology

This project is broken into three phases:

1. **Baseline System (FP32):**

   * Fine-tune a MobileNetV3-Small model on the CIFAR-100 dataset.

   * Split the model into an `edge_model` and `server_model`.

   * Implement and train a `Decoder` (attacker) network to reconstruct images from the FP32 (32-bit) features.

   * Benchmark the baseline Accuracy, Latency, Model Size, and Privacy (using PSNR).

2. **Quantized System (INT8):**

   * Apply Post-Training Quantization (PTQ) to the `edge_model` to create an `int8_edge_model`.

   * This INT8 model is what would run on the edge device.

3. **Analysis & Comparison:**

   * Run the *same* `Decoder` attack on the new INT8 features.

   * Measure the drop in reconstruction quality (PSNR).

   * Benchmark the new INT8 system for Accuracy, Latency, and Model Size.

   * Create a final table comparing the FP32 vs. INT8 trade-offs.

### Tech Stack

* **Framework:** PyTorch

* **Quantization:** `torch.quantization` (PTQ)

* **Models:** `torchvision.models.mobilenet_v3_small`

* **Dataset:** CIFAR-100

* **Metrics:** scikit-image (`psnr`), `tqdm`, `matplotlib`

### Project Roadmap

* [ ] **Phase 1: Baseline System**

  * [ ] Fine-tune MobileNetV3-Small on CIFAR-100.

  * [ ] Implement `baseline_setup.py` to split the model.

  * [ ] Implement `attacker.py` to train the decoder.

  * [ ] Get baseline FP32 PSNR, accuracy, and latency.

* [ ] **Phase 2: Quantization**

  * [ ] Implement `quantize.py` using `torch.quantization`.

  * [ ] Calibrate and convert the `edge_model` to INT8.

* [ ] **Phase 3: Evaluation**

  * [ ] Run INT8 model through the full pipeline.

  * [ ] Measure INT8 PSNR, accuracy, and latency.

* [ ] **Phase 4: Final Report**

  * [ ] Generate comparison plots and the final "trade-off" table.

  * [ ] Write up final analysis and conclusions.

### How to Run

1. Clone the repository:

`git clone https://github.com/your-username/your-repo-name.git cd your-repo-name`

2. Install dependencies:

`pip install torch torchvision scikit-image matplotlib tqdm`

3. Run the baseline and attack training:

`python baseline_setup.py python attacker.py`

4. Run the quantization and final evaluation:

`python quantize.py`
