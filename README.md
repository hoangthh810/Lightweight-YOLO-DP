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

## References

This project is built upon and improves the baseline architecture proposed in the following paper:

* **Paper Title:** PDT: Uav Target Detection Dataset for Pests and Diseases Tree
* **Authors:** Mingle Zhou, Rui Xing, Delong Han, Zhiyong Qi, Gang Li
* **Preprint:** [arXiv:2409.15679](https://arxiv.org/pdf/2409.15679)


