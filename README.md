# PrIM (Processing-In-Memory Benchmarks)

PrIM is the first benchmark suite for a real-world processing-in-memory (PIM) architecture.
PrIM is developed to evaluate, analyze, and characterize the first publicly-available real-world processing-in-memory (PIM) architecture, the [UPMEM](https://www.upmem.com/) PIM architecture.
The UPMEM PIM architecture combines traditional DRAM memory arrays with general-purpose in-order cores, called DRAM Processing Units (DPUs), integrated in the same chip.

PrIM provides a common set of workloads to evaluate the UPMEM PIM architecture with and can be useful for programming, architecture and system researchers all alike to improve multiple aspects of future PIM hardware and software.
The workloads have different characteristics, exhibiting heterogeneity in their memory access patterns, operations and data types, and communication patterns.
This repository also contains baseline CPU and GPU implementations of PrIM benchmarks for comparison purposes.

PrIM also includes a set of microbenchmarks can be used to assess various architecture limits such as compute throughput and memory bandwidth.

## Citation

Please cite the following paper if you find this repository useful:

Juan Gómez-Luna, Izzat El Hajj, Ivan Fernandez, Christina Giannoula, Geraldo F. Oliveira, and Onur Mutlu, "[Benchmarking a New Paradigm: An Experimental Analysis of a Real Processing-in-Memory Architecture](https://arxiv.org/pdf/2105.03814.pdf)". arXiv:2105.03814 [cs.AR], 2021.

Bibtex entry for citation:

```
@misc{gomezluna2021prim,
  title={{Benchmarking a New Paradigm: An Experimental Analysis of a Real Processing-in-Memory Architecture}},
  author={Juan Gómez-Luna and Izzat El Hajj and Ivan Fernandez and Christina Giannoula and Geraldo F. Oliveira and Onur Mutlu},
  year={2021},
  eprint={2105.03814},
  archivePrefix={arXiv},
  primaryClass={cs.AR}
}
```

## Repository Structure and Installation

We point out next the repository structure and some important folders and files.
All benchmark folders have similar structure to the one shown for BFS.
The microbenchmark folder contains eight different microbenchmarks, each with similar folder structure.
The repository also includes `run_*.py` scripts to run strong and weak scaling experiments for PrIM benchmarks.

```
.
+-- LICENSE
+-- README.md
+-- run_strong_full.py
+-- run_strong_rank.py
+-- run_weak.py
+-- BFS/
|   +-- baselines/
|	|	+-- cpu/
|	|	+-- gpu/
|   +-- data/
|   +-- dpu/
|   +-- host/
|   +-- support/
|   +-- Makefile
+-- BS/
|   +-- ...
+-- GEMV/
|   +-- ...
+-- HST-L/
|   +-- ...
+-- HST-S/
|   +-- ...
+-- MLP/
|   +-- ...
+-- Microbenchmarks/
|   +-- Arithmetic-Throughput/
|   +-- CPU-DPU/
|   +-- MRAM-Latency/
|   +-- Operational-Intensity/
|   +-- Random-GUPS/
|   +-- STREAM/
|   +-- STRIDED/
|   +-- WRAM/
+-- NW/
|   +-- ...
+-- RED/
|   +-- ...
+-- SCAN-SSA/
|   +-- ...
+-- SCAN-RSS/
|   +-- ...
+-- SEL/
|   +-- ...
+-- SpMV/
|   +-- ...
+-- TRNS/
|   +-- ...
+-- TS/
|   +-- ...
+-- UNI/
|   +-- ...
+-- VA/
|   +-- ...
```

### Prerequisites

Running PrIM requires installing the [UPMEM SDK](https://sdk.upmem.com).
PrIM benchmarks and microbenchmarks are designed to run on a server with real UPMEM modules, but they also run on the functional simulator include in the UPMEM SDK.

### Getting Started

Clone the repository:

```sh
git clone https://github.com/CMU-SAFARI/prim-benchmarks

cd prim-benchmarks
```

## Running PrIM

### PrIM Benchmarks

The repository includes scripts to run weak scaling and strong scaling experiments:

- `run_weak.py`: Weak scaling experiments for 16 PrIM benchmarks using 1 rank of UPMEM DPUs (1 to 64 DPUs).
- `run_strong_rank.py`: Strong scaling experiments for 16 PrIM benchmarks using 1 rank of UPMEM DPUs (1 to 64 DPUs).
- `run_strong_full.py`: Strong scaling experiments for 16 PrIM benchmarks using 4 to 32 ranks of UPMEM DPUs (256 to 2048 DPUs).

To use these scripts, update `rootdir` in the beginning of each script. To run weak scaling experiments for BFS or SpMV, update the paths to input files in `run_weak.py`.
The scripts save the results in a folder called `profile` inside each benchmark folder.

```sh
cd prim-benchmarks

# Weak scaling experiments for BFS
python3 run_weak.py BFS
```

Inside each PrIM benchmark folder, one can compile and run each benchmark with different input parameters.
Choose a benchmark and compile. Every Makefile accepts several input parameters:

```sh
cd BFS

# Compile BFS for 32 DPUs and 16 tasklets (i.e., software threads) per DPU
NR_DPUS=32 NR_TASKLETS=16 make all
```

For help instructions:

```sh
./bin/host_code -h
```

Run the benchmark:

```sh
./bin/host_code -v 0 -f data/loc-gowalla_edges.txt
```

Several benchmark folders (HST-S, HST-L, RED, SCAN-SSA, SCAN-RSS) contain a script (`run.sh`) that compiles and runs the benchmark for the experiments in the appendix of the [paper](https://arxiv.org/pdf/2105.03814.pdf).

### Microbenchmarks

Each microbenchmark folder contains a script (`run.sh`) that compiles and runs the microbenchmark for the experiments in the [paper](https://arxiv.org/pdf/2105.03814.pdf):

```sh
cd Microbenchmarks/Arithmetic-Throughput

./run.sh
```

### Getting Help

If you have any suggestions for improvement, please contact el1goluj at gmail dot com.
If you find any bugs or have further questions or requests, please post an issue at the [issue page](https://github.com/CMU-SAFARI/prim-benchmarks/issues).

## Acknowledgments

We thank UPMEM’s Fabrice Devaux, Rémy Cimadomo, Romaric Jodin, and Vincent Palatin for their valuable support. We acknowledge the support of SAFARI Research Group’s industrial partners, especially ASML, Facebook, Google, Huawei, Intel, Microsoft, VMware, and the Semiconductor Research Corporation. Izzat El Hajj acknowledges the support of the University Research Board of the American University of Beirut (URB-AUB-103951-25960).
