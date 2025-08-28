# IBAMR Eel Motion Simulation Project

A computational fluid dynamics project based on the IBAMR (Immersed Boundary Adaptive Mesh Refinement) framework for simulating eel swimming kinematics and hydrodynamics.

## Project Overview

This project implements:
- Eel swimming kinematics simulation (IBEELKinematics)
- Cylinder motion kinematics simulation (IBCylinderKinematics)
- Python-based reinforcement learning agent (PPO Agent)
- Distributed parallel computing support
- Automatic restart and monitoring mechanisms

## HPC Platform

This project is designed to run on high-performance computing (HPC) systems provided by [Paratera Cloud](https://cloud.paratera.com/). The platform offers:
- High-performance computing clusters
- MPI parallel computing infrastructure
- Slurm job scheduling system
- Optimized scientific computing environment

## System Requirements

- **Operating System**: Linux (optimized for Paratera Cloud HPC environment)
- **Compiler**: GCC 12.2+
- **MPI**: MPI parallel computing support
- **Job Scheduler**: Slurm workload manager
- **Dependencies**:
  - IBAMR 0.16.0
  - PETSc 3.22.2
  - SAMRAI 2025.01.09
  - nlohmann_json 3.7.0+
  - Python 3.x + PyTorch

## Quick Start

### 1. Environment Setup

Upload all project files to the same directory on your Paratera Cloud HPC system:

```bash
# Ensure the following files are in the same directory:
# - build_envs.sh
# - run_envs.sh
# - CMakeLists.txt
# - main.cpp
# - IBEELKinematics.cpp/h
# - IBCylinderKinematics.cpp/h
# - input2d
# - cylinder2d.vertex
# - start_files_np12/ (contains various parameter combinations)
# - model_server.py
# - ppg_agent.py
# - Other Python scripts
```

### 2. Build Parallel Computing Environments

Run the build script to create and compile all simulation environments:

```bash
chmod +x build_envs.sh
./build_envs.sh
```

**Script Functionality**:
- Automatically scans parameter combinations in `start_files_np12/` directory
- Creates independent directories for each environment (env1, env2, ..., env5)
- Copies corresponding configuration files and restart files
- Compiles using CMake to generate executable `main2d`
- Generates detailed environment allocation logs

**Environment Configuration**:
- Default creation of 5 parallel environments
- Each environment contains independent configuration files and restart states
- Automatic handling of parameter combination allocation

### 3. Submit HPC Jobs

Use Slurm job scheduling system to submit parallel computing tasks on Paratera Cloud:

```bash
chmod +x run_envs.sh
sbatch run_envs.sh
```

**Script Functionality**:
- Automatic allocation of compute nodes
- Parallel launch of multiple simulation environments
- Built-in monitoring and automatic restart mechanisms
- Support for failure detection and success state management

**Slurm Configuration**:
- Partition: `amd_256`
- Nodes: 1
- Total processes: 64
- MPI processes per environment: 12

## Project Structure

```
Project Root/
├── build_envs.sh          # Environment build script
├── run_envs.sh            # Job submission script
├── CMakeLists.txt         # CMake build configuration
├── main.cpp               # Main program entry
├── IBEELKinematics.cpp/h  # Eel kinematics implementation
├── IBCylinderKinematics.cpp/h # Cylinder kinematics implementation
├── input2d                # Simulation input parameters
├── cylinder2d.vertex      # Cylinder geometry file
├── start_files_np12/      # Parameter combination files
│   ├── x0_y0/            # Parameter combination 1
│   ├── x0_y0_c1/         # Parameter combination 2
│   └── ...               # More parameter combinations
├── model_server.py        # Python model server
├── ppg_agent.py          # PPG reinforcement learning agent
├── plot_utils.py          # Plotting utilities
└── state_processor.py     # State processor
```

## Core Features

### Simulation Engine
- **IBAMR Framework**: Immersed boundary method with adaptive mesh refinement
- **Parallel Computing**: MPI multi-process parallel simulation support
- **Adaptive Meshing**: Automatic optimization of computational grids for efficiency

### Intelligent Control
- **Python Interface**: Control simulation parameters through Python scripts
- **Reinforcement Learning**: Integrated PPG agent for motion optimization
- **Real-time Communication**: Socket-based real-time data exchange

### Monitoring & Management
- **Auto-restart**: Automatic restart after failures
- **State Monitoring**: Real-time monitoring of simulation progress and status
- **Log Management**: Detailed runtime logs and error records

## Usage Guide

### Basic Simulation Workflow

1. **Environment Build**: Run `build_envs.sh` to create and compile environments
2. **Job Submission**: Run `run_envs.sh` to submit HPC jobs on Paratera Cloud
3. **Runtime Monitoring**: Scripts automatically monitor and manage all environments
4. **Result Collection**: Collect result files after simulation completion

### Parameter Configuration

- Modify `input2d` file to adjust simulation parameters
- Add new parameter combinations in `start_files_np12/`
- Adjust environment count in `build_envs.sh`

### Custom Configuration

- Modify `CMakeLists.txt` to adjust compilation options
- Edit Slurm parameters in `run_envs.sh`
- Adjust agent parameters in Python scripts

## Log Files

- `allocation_log.txt`: Detailed environment allocation logs
- `env*.log`: Runtime logs for each environment
- `server.log`: Python server logs
- `model_server_log.txt`: Model server logs

## Dependencies

### Required Libraries
- **IBAMR**: Immersed boundary adaptive mesh refinement library
- **PETSc**: Portable, extensible toolkit for scientific computation
- **SAMRAI**: Structured adaptive mesh refinement application infrastructure
- **nlohmann/json**: Modern C++ JSON library

### Python Dependencies
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing library
- **Matplotlib**: Plotting library

---

**Note**: This project is specifically designed for HPC environments and has been tested on [Paratera Cloud](https://cloud.paratera.com/) infrastructure. It requires corresponding parallel computing infrastructure support.
