# Parametric Matrix Models (PMMs) — Research Notebooks

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange) ![JAX](https://img.shields.io/badge/JAX-autodiff-9cf) ![Qiskit](https://img.shields.io/badge/Qiskit-Aer-6929C4)

Undergraduate research (2025) on **Parametric Matrix Models**, a machine-learning approach in which outputs are obtained from eigenvalues/eigenvectors of learned matrices that depend on the input features (see Cook *et al.*, [arXiv:2401.11694](https://arxiv.org/abs/2401.11694)). These notebooks implement PMMs from scratch and explore their use for quantum-system emulation and noisy quantum hardware.

## Highlights

- PyTorch implementation of a PMM with symmetric parameter matrices `M(x) = M0 + x·M1`, trained by gradient descent.
- Affine **effective-Hamiltonian PMM** (JAX) that learns the ground-state energy E0(g) of the transverse-field Ising model.
- Affine **observable PMM** for a general classification task (2-D XOR), trained with complex-valued Adam.
- Gradient-based synthesis of 2-qubit unitaries (L-BFGS-B over 64 real parameters) and a Qiskit Bell-state pipeline.
- Zero-noise-extrapolation (ZNE) experiment combining a PMM with Qiskit Aer depolarizing noise models.

## Credentials

The IBM Quantum notebook reads its API token from the `IBM_QUANTUM_TOKEN` environment variable (see `.env.example`).

## Contents

- [`notebooks/01_pmm_pytorch_basics.ipynb`](notebooks/01_pmm_pytorch_basics.ipynb) — minimal PMM in PyTorch
- [`notebooks/02_pmm_hamiltonian_ising.ipynb`](notebooks/02_pmm_hamiltonian_ising.ipynb) — affine Hamiltonian PMM on the transverse-field Ising model
- [`notebooks/03_pmm_regression_xor.ipynb`](notebooks/03_pmm_regression_xor.ipynb) — affine observable PMM on XOR
- [`notebooks/04_unitary_optimization.ipynb`](notebooks/04_unitary_optimization.ipynb) — gradient-based 2-qubit unitary synthesis
- [`notebooks/05_qiskit_bell_state.ipynb`](notebooks/05_qiskit_bell_state.ipynb) — Qiskit / IBM Runtime Bell-state workflow
- [`notebooks/06_zne_qiskit.ipynb`](notebooks/06_zne_qiskit.ipynb) — zero-noise extrapolation with a PMM

## Repository Structure

```text
parametric-matrix-models/
├── notebooks/
│   ├── 01_pmm_pytorch_basics.ipynb
│   ├── 02_pmm_hamiltonian_ising.ipynb
│   ├── 03_pmm_regression_xor.ipynb
│   ├── 04_unitary_optimization.ipynb
│   ├── 05_qiskit_bell_state.ipynb
│   └── 06_zne_qiskit.ipynb
├── .env.example
├── README.md
└── requirements.txt
```

## Tech Stack

Python, JAX, PyTorch, NumPy, SciPy, Qiskit, Qiskit Aer, Qiskit IBM Runtime

## Getting Started

```bash
git clone https://github.com/siddakmarwaha/parametric-matrix-models.git
cd parametric-matrix-models
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

## Author

**Siddak Marwaha**

## License

All rights reserved. This is undergraduate research work shared for portfolio review; please contact the author before reusing code or results.
