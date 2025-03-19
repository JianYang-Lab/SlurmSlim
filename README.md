# SlurmSlim 💵

**Optimize Slurm Job Scheduling with Intelligent Memory Estimation**

## Overview
SlurmSlim is a lightweight and efficient tool designed to optimize job scheduling in **Slurm** by accurately estimating the required memory for scripts and programs. By leveraging **Model Context Protocol (MCP)**, **LLM models**, **file sizes**, and **system information**, SlurmSlim helps reduce computing costs by preventing over-allocated memory requests.

## Features
- ✅ **Intelligent Memory Estimation** – Uses MCP and LLM models to predict the optimal memory allocation.
- ✅ **Cost Reduction** – Prevents excessive memory requests, lowering overall compute costs.
- ✅ **File & System-Aware** – Considers file sizes and system specs for precise estimation.
- ✅ **Lightweight & Fast** – Designed for efficiency with minimal overhead.

## Installation
```sh
git clone https://github.com/JianYang-Lab/SlurmSlim.git
cd SlurmSlim
uv sync  # If applicable
```

## Usage
```sh
uv run client.py server.py
```


## Example Output
```
Estimated Memory: 8.2 GB
Suggested Slurm Command: sbatch --mem=8500M job_script.sh
```

## Why Use SlurmSlim?
- 🔹 **Saves Money** – No more over-provisioning, reducing unnecessary cloud or HPC costs.
- 🔹 **Improves Efficiency** – Ensures jobs run smoothly without excessive memory requests.
- 🔹 **Seamless Integration** – Works directly with Slurm job scripts and scheduling workflows.

## Future Work
- Extend support for CPU & GPU resource optimization
- Integration with other job schedulers (e.g., PBS, LSF)
- Advanced machine learning models for prediction

## License
📜 MIT License

## Contributors

- [Wenjie Wei](https://github.com/wjwei-handsome)
- [Lounan Li](https://github.com/SGGb0nd)