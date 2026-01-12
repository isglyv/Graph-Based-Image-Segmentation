# 👁️ GraphVision: Graph-Based Image Segmentation

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Library](https://img.shields.io/badge/NetworkX-Graph%20Theory-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> **"Seeing beyond pixels: Understanding images through Complex Networks."**

## 📖 Overview

**GraphVision** is a computer vision project developed for the **Complex Networks** course. It implements **Graph-Based Image Segmentation** using **Spectral Clustering** and **Normalized Cuts (N-Cuts)**.

Unlike traditional methods that treat images as simple grids of pixels, this project models the image as a **Weighted Undirected Graph**. It leverages **SLIC Superpixels** for optimization, making it possible to process high-resolution images efficiently while preserving spatial coherence.

---

## 🚀 Key Features

* **Graph Construction:** Converts images into Region Adjacency Graphs (RAG).
* **Spectral Clustering:** Uses the Laplacian Matrix and Eigenvalue decomposition to partition the image.
* **Optimization:** Implements **SLIC (Simple Linear Iterative Clustering)** to reduce computational complexity from $O(N^2)$ to manageable levels.
* **Benchmarking:** Includes a comparison module against the classic **K-Means** clustering algorithm.

---

## 📊 Results & Analysis

We compared the **Graph-Based** approach against **K-Means**.

| Feature | K-Means (Traditional) | Graph-Based (Our Approach) |
| :--- | :--- | :--- |
| **Methodology** | Color Clustering only | Color + Spatial Connectivity |
| **Noise Sensitivity** | High (Salt-and-pepper noise) | Low (Coherent regions) |
| **Edge Preservation** | Weak | Strong |
| **Processing Time** | Fast (~0.4s) | Moderate (~1.5s - 5s) |

### Visual Comparison

<img width="1466" height="493" alt="Ekran Resmi 2026-01-12 11 37 48" src="https://github.com/user-attachments/assets/bf993732-6d2c-4376-8ddf-5cb0c3eb5214" />

> **Observation:** As seen above, K-Means fails to distinguish shadows from objects, creating noise. The Graph-Based method preserves object integrity (the cup and the plate) as unified regions.

---

## 🧪 Mathematical Background

The project relies on the **Normalized Cut** criterion proposed by Shi & Malik.

**1. Affinity Matrix ($W$):**
We calculate the weight between two nodes $i$ and $j$ based on color and spatial proximity:
$$w_{ij} = e^{\frac{-||F_i - F_j||^2}{\sigma_I^2}} \cdot e^{\frac{-||X_i - X_j||^2}{\sigma_X^2}}$$

**2. Laplacian Matrix:**
To solve the graph partitioning problem, we compute the unnormalized Laplacian:
$$L = D - W$$

The segmentation is then derived from the **Eigenvectors** of this matrix (Spectral Clustering).

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/isglyv/GraphVision.git](https://github.com/isglyv/GraphVision.git)
    cd GraphVision
    ```

2.  **Create a Virtual Environment (Optional but recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 💻 Usage

To run the segmentation analysis and see the comparison:

```bash
python main.py
