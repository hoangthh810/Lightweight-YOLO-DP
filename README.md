## Abstract

Autonomous UAV-based plant protection operations require high-precision object detection models that can effectively identify dense, small-scale targets such as agricultural pests, diseases, and weeds in real-time. While the advanced YOLO-DP baseline (proposed in *"PDT: Uav Target Detection Dataset for Pests and Diseases Tree"*) achieves outstanding accuracy, its high computational complexity poses significant challenges for deployment on resource-constrained edge devices like NVIDIA Jetson or Raspberry Pi.

To bridge this gap, this project focuses on hardware-efficient architectural optimization to compress the network into a **Tiny variant**. By integrating **Low-Rank Approximation** with **GhostConv** and **Depthwise Separable Convolution**, we drastically eliminate both inter-channel and spatial computational redundancies—minimizing GFLOPs and total parameters while maintaining high detection performance.

Our optimized model delivers an optimal balance between latency, model size, and accuracy directly on edge computing hardware.


## Download Dataset
Hugging Face: PDT dataset v2 (Improve the quality 2024.10.4)

## Code
GitHub: https://github.com/hoangthh810/Lightweight-YOLO-DP

## References

This project is built upon and improves the baseline architecture proposed in the following paper:

* **Paper Title:** PDT: Uav Target Detection Dataset for Pests and Diseases Tree
* **Authors:** Mingle Zhou, Rui Xing, Delong Han, Zhiyong Qi, Gang Li
* **Preprint:** [arXiv:2409.15679](https://arxiv.org/pdf/2409.15679)
