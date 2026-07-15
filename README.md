## Abstract

This project presents a lightweight pest and disease detection model designed for deployment on resource-constrained edge devices, particularly the Raspberry Pi 4. The original YOLO-DP architecture was adapted for low-altitude UAV imagery using the PDT (LL) dataset. The optimization process includes removing the computationally expensive Large-Scale Selective Kernel module, pruning the P5 detection branch, simplifying the backbone, and integrating lightweight components such as SCDown, UIBLite, low-rank convolution, and ReLU6 activation. The optimized YOLO-UIBLite-LowRank model contains only 0.36 million parameters and requires 4.3 GFLOPs, reducing the parameter count by 94.1% and computational complexity by 66.4% compared with YOLO-DP-21, while maintaining an FP32 mAP@0.5 of 0.911. For edge deployment, the model was exported to ONNX and quantized to INT8 using the QOperator format. On a Raspberry Pi 4 Model B, the final model achieved an average processing time of 199 ms per image, equivalent to 5.03 FPS. It obtained a precision of 0.8617, recall of 0.8479, mAP@0.5 of 0.8816, and mAP@0.5:0.95 of 0.5513. These results demonstrate an effective balance between detection accuracy, computational efficiency, and practical edge deployment for UAV-based agricultural monitoring.


## Download Dataset
**Hugging Face:** [PDT dataset v2 (Improve the quality 2024.10.4)](https://huggingface.co/)

## Code
**GitHub:** [YOLO-DP-Tiny](https://github.com/hoangthh810/Lightweight-YOLO-DP)

## Datasets
### PDT dataset

**Class:** unhealthy

![Target Examples](img/target.png)

(a) is a healthy goal and (b) is an unhealthy goal. The PDT dataset takes (b) as the category.

**Double Resolution:**

![Double Resolution Dataset](img/PDT_LL_LH.png)

## Dataset Structure

| Edition | Classes | Structure | Targeted images | Untargeted images | Image size | Instances | Target Amount <br> S(Small) <br> M(Medium) <br> L(Large) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sample** | unhealthy | Train | 81 | 1 | $640 \times 640$ | 2569 | 1896 548 179 |
| | | Val | 19 | 1 | $640 \times 640$ | 691 | 528 138 25 |
| **LL** | unhealthy | Train | 3166 | 1370 | $640 \times 640$ | 90290 | 70418 16342 <br> 3530 |
| | | Val | 395 | 172 | $640 \times 640$ | 12523 | 9926 2165 432 |
| | | Test | 390 | 177 | $640 \times 640$ | 11494 | 8949 2095 450 |
| **LH** | unhealthy | - | 105 | 0 | $5472 \times 3648$ | 93474 | 93474 0 0 |

## Models

### Network Structure

![Network Structure](img/model.png)

## Experiments

### Experimental Setup

All model variants were trained and evaluated under the same dataset and hyperparameter settings to ensure a fair comparison.

| Configuration | Description |
|---|---|
| Dataset | PDT (LL), containing low-altitude UAV images of pine trees affected by Red Turpentine Beetle |
| Input resolution | 640 × 640 pixels |
| Training platform | Google Colab |
| Training GPU | NVIDIA Tesla T4 |
| Training epochs | 60 epochs for each architecture |
| GPU benchmark device | NVIDIA GeForce RTX 4060 |
| GPU benchmark protocol | Each model was tested five times, and the highest FPS was reported |
| Edge device | Raspberry Pi 4 Model B |
| Edge processor | Broadcom BCM2711, Quad-core ARM Cortex-A72 at 1.5 GHz |
| Memory | 8 GB LPDDR4 |
| Inference engine | ONNX Runtime CPU Execution Provider |
| Quantization | INT8 Conv-only quantization using QDQ and QOperator formats |
| Edge benchmark protocol | Five runs per configuration, with the average latency reported |
| Confidence threshold | 0.45 |
| IoU threshold | 0.45 |
| Test set | 567 images with 11,598 ground-truth objects |

### FP32 Architecture Comparison

The following table compares the original YOLO-DP architectures and the proposed lightweight variants before INT8 quantization.

| Model | Layers | Parameters | GFLOPs | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO-DP-21 | 129 | 6.10M | 12.8 | 0.888 | 0.866 | 0.934 | 0.645 | 242 |
| YOLO-DP-K23 | 96 | 3.40M | 9.5 | **0.889** | **0.867** | **0.938** | **0.653** | 285 |
| YOLO-DP-K23-Tiny | 177 | 0.57M | 3.2 | 0.863 | 0.839 | 0.916 | 0.600 | 185 |
| YOLO-ACNET | 131 | **0.20M** | **2.6** | 0.856 | 0.848 | 0.918 | 0.598 | 260 |
| YOLO-DP-Depthwise-Lowrank | 128 | **0.20M** | **2.6** | 0.864 | 0.835 | 0.915 | 0.600 | 250 |
| YOLO-MobileOne-PConv | 103 | 0.38M | 3.7 | 0.873 | 0.838 | 0.920 | 0.607 | 290 |
| YOLO-RELAN-SCDown | 96 | 0.32M | 4.1 | 0.873 | 0.849 | 0.923 | 0.613 | 303 |
| **YOLO-UIBLite-Lowrank** | **85** | **0.36M** | **4.3** | **0.858** | **0.840** | **0.911** | **0.582** | **303** |
| YOLO-UIBLite-NoLowrank | **76** | 0.69M | 5.9 | 0.873 | 0.840 | 0.919 | 0.596 | **312** |

### Raspberry Pi 4 Inference Performance

Three lightweight models were exported to ONNX and quantized to INT8. Both QDQ and QOperator quantization formats were evaluated directly on the Raspberry Pi 4.

| Model | INT8 Format | Preprocessing | Inference | Total Time | FPS |
|---|---|---:|---:|---:|---:|
| **YOLO-UIBLite-Lowrank** | QDQ | 10.61 ms | 187.27 ms | 201.29 ms | 4.97 |
| **YOLO-UIBLite-Lowrank** | **QOperator** | **9.27 ms** | **186.40 ms** | **199.00 ms** | **5.03** |
| YOLO-RELAN-SCDown | QDQ | 10.36 ms | 196.03 ms | 209.69 ms | 4.77 |
| YOLO-RELAN-SCDown | QOperator | 8.91 ms | 193.16 ms | 205.31 ms | 4.87 |
| YOLO-UIBLite-NoLowrank | QDQ | 9.70 ms | 212.10 ms | 225.02 ms | 4.44 |
| YOLO-UIBLite-NoLowrank | QOperator | 9.24 ms | 221.18 ms | 233.62 ms | 4.28 |


### INT8 Detection Accuracy

The quantized models were evaluated on a test set containing 567 images and 11,598 ground-truth objects.

| Model | INT8 Format | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---:|---:|---:|---:|
| YOLO-UIBLite-Lowrank | QDQ | 0.8622 | **0.8483** | **0.8824** | 0.5517 |
| **YOLO-UIBLite-Lowrank** | **QOperator** | 0.8617 | 0.8479 | 0.8816 | 0.5513 |
| YOLO-UIBLite-NoLowrank | QDQ | 0.8907 | 0.8372 | **0.8852** | 0.5819 |
| YOLO-UIBLite-NoLowrank | QOperator | 0.8922 | 0.8353 | 0.8847 | 0.5823 |
| YOLO-RELAN-SCDown | QDQ | 0.8980 | 0.8147 | 0.8737 | **0.5932** |
| YOLO-RELAN-SCDown | QOperator | **0.8987** | 0.8153 | 0.8740 | 0.5931 |

## Qualitative Results

To qualitatively evaluate the optimized model, the final
**YOLO-UIBLite-Lowrank INT8 QOperator** model was deployed on the
Raspberry Pi 4 and tested using images from the PDT (LL) test set.

The visualization uses the following bounding-box convention:

- **Green:** True Positive (TP)
- **Red:** False Positive (FP)
- **Blue:** Missed Ground Truth (MISS)

### Case 1: Dense Pest Distribution

This case evaluates the model under a challenging scenario containing
a high density of infected regions. Multiple targets appear close to
one another, making localization more difficult and increasing the
possibility of overlapping detections.

| Original Image | Prediction vs. Ground Truth |
|:---:|:---:|
| <img src="img/case1_dense_original.jpg" width="420"> | <img src="img/case1_dense_result.jpg" width="420"> |

**Detection summary:** `TP: 13`, `FP: 4`, `MISS: 4`

The model successfully detected most infected regions despite the high
target density. However, several false positives and missed objects
remained, indicating that densely distributed and partially overlapping
targets are still challenging for the lightweight INT8 model.

---

### Case 2: Sparse Pest Distribution

This case contains only a small number of infected regions distributed
across the UAV image. The targets are spatially separated and have
relatively clear visual boundaries.

| Original Image | Prediction vs. Ground Truth |
|:---:|:---:|
| <img src="img/case2_sparse_original.jpg" width="420"> | <img src="img/case2_sparse_result.jpg" width="420"> |

**Detection summary:** `TP: 3`, `FP: 0`, `MISS: 0`

All ground-truth objects were correctly detected without false positives
or missed detections. This result demonstrates that the model performs
reliably when infected regions are sparse and visually distinguishable
from the surrounding vegetation.

---

### Case 3: No Pest or Disease Present

This negative case evaluates whether the model incorrectly generates
detections when no infected region is present in the image.

| Original Image | Model Output |
|:---:|:---:|
| <img src="img/case3_no_pest_original.jpg" width="420"> | <img src="img/case3_no_pest_result.jpg" width="420"> |

**Detection summary:** `TP: 0`, `FP: 0`, `MISS: 0`

The model produced no bounding boxes for this image, correctly identifying
the scene as containing no visible pest or disease regions. This result is
important for practical deployment because it demonstrates the model's
ability to avoid unnecessary false alarms in healthy forest areas.

### Overall Observation

The qualitative results show that the deployed model performs well in
sparse-target and negative-image scenarios. Its main limitations appear
in dense scenes, where small, overlapping, or visually ambiguous targets
may lead to false positives and missed detections. Nevertheless, the
model maintains a practical balance between detection quality and
computational efficiency for low-frame-rate UAV monitoring on the
Raspberry Pi 4.


## References

This project is built upon and improves the baseline architecture proposed in the following paper:

* **Paper Title:** PDT: Uav Target Detection Dataset for Pests and Diseases Tree
* **Authors:** Mingle Zhou, Rui Xing, Delong Han, Zhiyong Qi, Gang Li
* **Preprint:** [arXiv:2409.15679](https://arxiv.org/pdf/2409.15679)


