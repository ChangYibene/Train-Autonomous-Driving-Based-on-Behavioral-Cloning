README

# High-Speed Train Autonomous Driving Platform based on Behavioral Cloning

## 📖 Introduction

This project is a simulation platform for High-Speed Train Automatic Train Operation (ATO), developed as a graduation design. It explores a **"Teacher-Student"** control paradigm to balance control precision and real-time computational efficiency.

* **Teacher (Expert):** A Model Predictive Controller (MPC) that handles complex constraints (ATP limits, comfort, energy efficiency) but requires high computational cost.
* **Student (Learner):** A Deep Neural Network trained via **Behavioral Cloning (BC)**. It mimics the expert's strategy but operates with millisecond-level inference speed, making it suitable for embedded deployment.

The platform simulates the longitudinal dynamics of the **CR400AF Fuxing (复兴号)** high-speed train, featuring realistic constant-power traction characteristics and aerodynamic resistance models.

## ✨ Key Features

* **High-Fidelity Dynamics Model:** Simulates CR400AF train characteristics, including constant torque/power regions, Davis resistance equation, and actuator delays (first-order lag).
* **Modular Architecture:** Decoupled design for Physics, Control (MPC/AI), Environment, and GUI.
* **Interactive GUI:** Built with **PyQt5**, featuring real-time plotting of speed-distance curves, customizable track speed limits, and acceleration monitoring.
* **MPC Expert Controller:** Implements a constrained optimization logic to generate optimal driving trajectories (safety + comfort + efficiency).
* **End-to-End Learning:** Uses **PyTorch** to train a policy network that maps train states directly to control notches.

## 📂 Project Structure

```text
HighSpeedTrain_ATO/
│
├── main.py                  # Entry point of the application
├── gui_app.py               # PyQt5 GUI logic and visualization
├── train_dynamics.py        # CR400AF longitudinal dynamics model
├── track_profile.py         # Track generation and ATP curve calculation
├── mpc_controller.py        # Model Predictive Control (Expert) logic
├── policy_network.py        # PyTorch Neural Network definition
│
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

```

## 🚀 Installation & Requirements

### Prerequisites

* Python 3.8 or higher
* Anaconda (Recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/HighSpeedTrain_ATO.git
cd HighSpeedTrain_ATO

```


2. Install dependencies:
```bash
pip install numpy matplotlib PyQt5 torch

```


*(Note: If using Anaconda, ensure `pyqt` is properly installed via conda or pip)*

## 🛠️ Usage Guide

Run the main program to launch the simulation platform:

```bash
python main.py

```

### Workflow

1. **Configure Track (Optional):**
* In the left panel, modify the **ATP Speed Limits** (Start, End, Speed).
* Click **"Generate Curve"** to update the safety braking profile (Red Line).


2. **Step 1: Run MPC (Data Generation)**
* Click **"Start MPC"**.
* The train will drive automatically using the optimization algorithm.
* **Goal:** Collect high-quality "State-Action" pairs as training data.


3. **Step 2: Train Model**
* After the MPC run finishes, click **"Train Neural Network"**.
* The system will use the collected data to train the Policy Network (MLP) via Supervised Learning.
* Watch the Loss value decrease in the status bar.


4. **Step 3: Run AI (Verification)**
* Click **"Start AI"**.
* The Neural Network will take over control.
* **Goal:** Verify if the AI can reproduce the smooth and safe driving behavior of the MPC expert.



## ⚙️ Technical Details

### 1. Dynamics Model (CR400AF)

The simulation uses a single-mass point model normalized for a CR400AF carriage (approx. 55t).

* **Traction:** Limited by both maximum tractive force (28kN) and maximum power (1300kW). .
* **Resistance:** Davis Formula .
* **Braking:** Includes pneumatic braking delay simulated by a first-order inertial link ().

### 2. Model Predictive Control (MPC)

* **Prediction Horizon:** 3.0 seconds (30 steps).
* **Cost Function:** .
* **Constraints:** Hard constraints on speed (ATP limit) and control input (Throttle [-1, 1]).

### 3. Policy Network

* **Input:** Normalized Speed, Speed Error.
* **Architecture:** Fully Connected Network (MLP) with 3 hidden layers (64-128-64 units).
* **Activation:** ReLU for hidden layers, Tanh for output layer (to bound output between -1 and 1).

## 📚 References

This project is inspired by the following academic theses:

1. **Ning Chenhe.** *Automatic Train Operating Algorithm of High-Speed Train Based on Attention Mechanism and State Awareness*. Beijing Jiaotong University, 2022. (UI Design & Attention Mechanism)
2. **Zhao Wentao.** *Research on Modeling and Braking Characteristics of Air Braking System in a Train*. Southwest Jiaotong University, 2020. (Braking Model)
3. **Zhang Miao.** *Research on Automatic Train Operation based on Reinforcement Learning*. China Academy of Railway Sciences, 2020. (RL & DQN Concepts)

## 📝 License

This project is open-sourced under the MIT License.
Designed for academic and educational purposes only.

---

**Author:** Xiang Guanhua
**University:** BJTU School of Automation and Intelligence
**Date:** 2025.12.31

