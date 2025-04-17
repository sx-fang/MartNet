# Soc-MartNet for Parabolic Equations and HJB Equations

This repository contains the source code for the paper: **SOC-MartNet: A Martingale Neural Network for the Hamilton-Jacobi-Bellman Equation without Explicit $\inf_{u \in U} H$ in Stochastic Optimal Controls**.

The paper is authored by Wei Cai, Shuixin Fang, and Tao Zhou, and has been accepted for publication in the SIAM Journal on Scientific Computing.

The preprint of the paper is available on arXiv: [https://arxiv.org/abs/2405.03169](https://arxiv.org/abs/2405.03169).

## Dependencies

The following dependencies are required to run the code:

- `matplotlib==3.10.1`
- `numpy==2.2.4`
- `pandas==2.2.3`
- `psutil==5.9.0`
- `torch==2.6.0`

To enable efficient code execution on XPU or CUDA devices, ensure that you have installed a CUDA-enabled or XPU-enabled version of PyTorch, along with the necessary hardware drivers.


## Description of Modules

- **`runtask.py`**: Serves as the main entry point for running the algorithm.
- **`martnetdf.py`**: Implements the core logic of the algorithm.
- **`exmeta.py`**: Defines the metaclass for PDEs and provides related utility functions.
- **`examples`**: Contains example implementations for various parabolic and HJB equations.
- **`default_config.ini`**: Specifies the default configuration parameters for the algorithm.
- **`savresult.py`**: Provides functions for plotting results and saving outputs as CSV files.
- **`taskmaker.py`**: Generates task files, which are stored in the `./taskfiles` directory.
  
To run the algorithm, execute:

```bash
python runtask.py
```

**Behavior of `runtask.py`:**  
`runtask.py` starts the training process using parameters from either `default_config.ini` or any `.ini` task files located in the `./taskfiles` directory. All results and outputs are saved in the `./outputs` folder.  
If no `.ini` task files are found in `./taskfiles`, `runtask.py` will automatically use `default_config.ini`. If `.ini` task files are present, `runtask.py` will execute the tasks defined in those files instead.

## Citation
```bibtex
@misc{cai2025socmartnetmartingaleneuralnetwork,
      title={SOC-MartNet: A Martingale Neural Network for the Hamilton-Jacobi-Bellman Equation without Explicit inf H in Stochastic Optimal Controls}, 
      author={Wei Cai and Shuixin Fang and Tao Zhou},
      year={2025},
      eprint={2405.03169},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2405.03169}, 
}

@misc{cai2024martingaledeeplearninghigh,
      title={Martingale deep learning for very high dimensional quasi-linear partial differential equations and stochastic optimal controls}, 
      author={Wei Cai and Shuixin Fang and Wenzhong Zhang and Tao Zhou},
      year={2024},
      eprint={2408.14395},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2408.14395}, 
}
```
