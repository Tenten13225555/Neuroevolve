# Neuroevolutionary Algorithms for Neural Architecture Search

## Overview

This project aims to leverage neuroevolutionary algorithms to optimize neural network architectures. By evolving the neural architectures, we intend to automatically discover efficient structures that outperform hand-crafted designs.

## Dependencies

- Python (>= 3.6)
- TensorFlow (>= 2.4.0)
- Keras
- NumPy

To install the dependencies, run the following commands:

```bash
pip install tensorflow
pip install keras
pip install numpy
```

## Features

1. **Neuroevolution Engine**: A custom-built engine that evolves neural network architectures based on a genetic algorithm.
2. **Fitness Evaluation**: A module to evaluate the fitness of each neural architecture using specified benchmarks 
3. **Population Management**: Manages the population of neural architectures, including mutation, crossover, and selection mechanisms.
4. **Visualization Tools**: Utilities to visualize the evolution process and the discovered architectures.

## Usage

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Run the main script to start the neuroevolution process:

    ```bash
    python main.py
    ```

## Output

1. Logs containing the fitness score of each architecture at each generation.
2. Visualization of the best-performing architectures.
3. Saved model files of the best-performing architectures.
4. A summary report detailing the evolution process and final results.

## Note

This project is for research purposes and is a work in progress. The current version is a proof-of-concept and should not be considered as a final product (STILL IN DEVELOPMENT).
