# Traffic Control with Multi-Agent Reinforcement Learning

This repository contains implementations of various Multi-Agent Reinforcement Learning (MARL) algorithms to control traffic light intersections. The algorithms involve standard DQN, Double DQN (DDQN), Dueling Double DQN (D3QN), and integration with Graph Attention Networks (GAT) and Prioritized Experience Replay (PER).

## Project Structure

- **`configs/`**: Contains CityFlow configuration files, JSON traffic flows, and roadnet topologies (e.g., `Intersections_1`, `Intersections_4`).
- **`environments/`** / **`gym-cityflow/`**: A custom wrapper integrating CityFlow simulator with OpenAI Gym.
- **`logs/`**: Raw training and execution logs.
- **`notebooks/`**: Jupyter Notebooks for training, experimenting, and evaluating specific agents (e.g., `main.ipynb`, `evaluate.ipynb`, `loop_ddqn_gat.ipynb`, etc.).
- **`results/`**: Evaluation metrics, JSON files containing statistics, and plots demonstrating performance (e.g., `evaluation_results/`, `stats/`).
- **`src/`**: Shared Python source code and scripts (e.g., `buildgraph.py`).
- **`weights/`**: Checkpoints and `.pth` weight files for different RL models (`dqn`, `ddqn`, `duel_ddqn`, `d3qn_per`, `duel_ddqn_gat`, and baseline `project_weights`).

## Prerequisites

- **Python**: Version 3.8 is highly recommended for compatibility with CityFlow and Gym.
- **CUDA**: Optional but recommended for faster PyTorch training.

You can set up this project either globally via Docker, or locally using a Python virtual environment.

---

## Method 1: Running via Docker (Recommended)

A Dockerfile is provided to create an isolated environment with all dependencies, including CUDA 11.8 support, pre-installed.

1. **Build the Docker Image**:
   Navigate to the repository root and run:
   ```bash
   docker build -t traffic-control-env .
   ```

2. **Run the Docker Container**:
   Run the container while mounting this repository to the `/app` directory inside the container:
   ```bash
   docker run -it --gpus all -v /absolute/path/to/TrafficControl:/app traffic-control-env
   ```
   *Note: Omit `--gpus all` if you do not have an NVIDIA GPU or the NVIDIA container toolkit installed.*

3. **Install the `gym-cityflow` Environment**:
   Once inside the container, you need to install the custom local Gym environment wrapper:
   ```bash
   cd /app/gym-cityflow
   pip install -e .
   ```

---

## Method 2: Local Setup (using pyenv / Virtual Environment)

If you prefer to run things locally without Docker, follow these steps to use `pyenv` and a virtual environment.

1. **Install Python 3.8 with pyenv** (if you don't have it installed):
   ```bash
   pyenv install 3.8
   pyenv local 3.8
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   
   # On Windows
   .\venv\Scripts\activate
   
   # On Linux/MacOS
   source venv/bin/activate
   ```

3. **Install Core Machine Learning Libraries**:
   Install PyTorch (with or without CUDA depending on your system), numpy, matplotlib, and torch-geometric:
   ```bash
   pip install numpy matplotlib pillow wheel setuptools==57.5.0
   
   # Example: Install PyTorch with CUDA 11.8 (Adjust the URL for your CUDA version or use CPU)
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   
   pip install torch-geometric gym==0.23.1
   ```

4. **Install CityFlow**:
   CityFlow requires C++ build tools (CMake, MSVC/GCC). Make sure you have those installed before running this.
   ```bash
   git clone https://github.com/cityflow-project/CityFlow.git
   cd CityFlow
   pip install .
   cd ..
   ```

5. **Install Local `gym-cityflow` Wrapper**:
   Navigate to the local wrapper directory and install it:
   ```bash
   cd gym-cityflow
   pip install -e .
   cd ..
   ```

---

## Running the Code

The different algorithm implementations and logic are accessible via Jupyter Notebooks located in the `notebooks/` directory.

You can run the evaluations by opening their respective Jupyter Notebooks (`.ipynb`), testing the main entry points, or training specific algorithm versions:
- `notebooks/main.ipynb` (Main experiments entry point)
- `notebooks/evaluate.ipynb` (Evaluation and metric plotting)
- `notebooks/noloop_ddqn_gat.ipynb`, `notebooks/d3qn_per.ipynb`, etc. (Algorithm-specific training)

To use Jupyter Notebooks, ensure you install Jupyter in your environment:
```bash
pip install jupyter
jupyter notebook
```


## Evaluation Results

The models were evaluated and compared across various metrics such as total arrived vehicles, active vehicles, average wait time, and average travel time. Below is a summary of the performance compared to baselines:

```text
==========================================================================================
Model                     Arrived    Active               Avg Wait (s)         Avg Travel (s)      
==========================================================================================
Fixed Time (Baseline)     2346       746                  51.64                184.78     
Standard DQN              2541       574 (-23.1%)         12.06 (-76.6%)       153.87 (-16.7%)     
Double DQN                2545       570 (-23.6%)         13.75 (-73.4%)       155.78 (-15.7%)     
D3QN                      2561       554 (-25.7%)         10.07 (-80.5%)       151.75 (-17.9%)     
GAT-D3QN                  2540       575 (-22.9%)         12.88 (-75.1%)       154.76 (-16.2%)     
PER-GAT-D3QN              2543       572 (-23.3%)         13.35 (-74.1%)       154.71 (-16.3%)     
==========================================================================================
```

**Key Takeaways**:
- **D3QN** performed the best overall, yielding an **80.5% reduction** in average wait times and a 17.9% reduction in average travel time over the fixed-time baseline.
- **Deep RL Agents** significantly outperformed the algorithmic non-RL baselines (Fixed Time and Max Pressure).
