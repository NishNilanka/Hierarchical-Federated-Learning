# Hierarchical Federated Learning (Client-Edge-Cloud)

This repository implements a **Hierarchical Federated Learning (HFL)** system across three tiers: **Clients**, **Edge Servers**, and a **Cloud Server**. The system leverages [Flower (FLWR)](https://flower.dev/) for federated learning simulations and enables efficient, privacy-preserving training of deep learning models using distributed data sources.

## 🧠 Project Overview

The architecture introduces a —**HierFAVG**—for hierarchical model aggregation:
- **Clients** perform local updates.
- **Edge servers** aggregate client updates periodically.
- **The cloud server** further aggregates edge models, reducing communication cost and improving model convergence.

This design offers better:
- **Communication efficiency**
- **Energy consumption trade-offs**
- **Training time performance**
- **Scalability across heterogeneous devices**

## 📂 Project Variants

This repository includes multiple HFL variants for different resource allocation strategies:

### 🔹 `HierFL_Dynamic_allocation.py`
Implements **dynamic client-to-edge allocation** at each global round based on real-time device metrics (e.g., energy, communication history). It improves fairness and robustness in federated settings with varying client capabilities.

### 🔹 `HierFL_Uniform_allocation.py`
Assigns clients **uniformly** to edge servers regardless of client-specific characteristics. This acts as a baseline strategy for evaluating more intelligent clustering or allocation policies.

## 🏗️ Components

- `client.py`: Client-side logic for local training and evaluation
- `device_manager.py`: Tracks client energy and training time
- `model.py`: Defines the deep learning model and training logic (MNIST/CIFAR-10)
- `strategy.py`: Custom aggregation strategy extending FedAvg
- `EdgeServer.py`: Abstraction for edge server logic
- `metrics.py`: Custom evaluation metrics
- `load_datasets.py`: Loads and partitions datasets (IID / non-IID)
- `utils.py`: Logging, cluster configuration, and random generation tools
- `main.py`: Main execution file with argument setup and training loop

## 📊 Datasets

- **MNIST** (default): Handwritten digits dataset


## 🚀 Getting Started

1. Clone the repository.
2. Install dependencies (`flwr`, `torch`, `sklearn`, etc.).
3. Modify `main.py` to use your desired FL variant:
   ```python
   from HierFL_Dynamic_allocation import HierFL
   ```
4. Run the simulation:
   ```bash
   python main.py
   ```

## 📈 Metrics Tracked

- Training accuracy and loss per round
- Energy consumed (computation + communication)
- Training time
- Number of communications

## 📎 Notes

- All energy and time metrics are simulated based on parameterized models.
- Experiment logs are saved in the `Experiments/` directory with timestamps.


