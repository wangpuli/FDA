## FDA: Flow Matching for Few-Trial Neural Adaptation with Stable Latent Dynamics

The repo is the official implementation for the paper: [Few-Trial Neural Adaptation with Stable Latent Dynamics](https://openreview.net/pdf?id=nKJEAQ6JCY).

### Overview

We introduce **Flow-Based Distribution Alignment (FDA)**, a novel framework that leverages flow matching to align neural activity across sessions with minimal data. The codebase is organized as follows:

``` framework
FDA/
├── align/        # Alignment loss functions (MMD and MLA)
├── condition/    # Conditional feature extractors
├── config/       # Model, pre-training, and alignment configurations
├── experiment/   # Pre-training and fine-tuning functions
├── flow/
│   ├── models/       # Flow-based backbone of FDA
│   └── transport/    # Diffusion and ODE-based sampling for flow matching
├── layers/       # Layers for conditional feature extractors
├── ldns/         # Simulated neural data generation (Lorenz attractor)
├── preprocess/   # Dataloader generation
├── utils/        # Utility functions (attention masks, spike preprocessing)
└── xds/          # Standard preprocessing for CO-C, CO-M, and RT-M datasets
```

### Quickstart

#### Data downloading

The CO-C, CO-M and RT-M datasets used here are available at [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.cvdncjt7n). They correspond to the following files:

- `Chewie_CO_2016.7z` (CO-C)  
- `Mihili_CO_2014.7z` (CO-M)  
- `Mihili_RT_2013_2014.7z` (RT-M)  

After downloading, extract the files and place them in the `dataset/` folder.  

#### Setup

We recommend setting up the environment using **conda** or **miniconda**. All required dependencies can be installed with:

```bash
git clone https://github.com/wangpuli/FDA.git
cd FDA
conda env create -f environment.yml
```

#### Training on simulated and real neural data

- **Simulated Neural Data**

    The simulated neural data is generated from the Lorenz attractor, following the implementation in [LDNS](https://github.com/mackelab/LDNS). Run FDA on the simulated neural data with:

    ```bash
    python test_FDA_on_simulated_neural_data.py
    ```


- **Spiking Recordings**

    The experimental scripts for pre-training and fine-tuning on the RT-M dataset can be run as follows. In this example, fine-tuning is performed with a target ratio of $r = 0.02$.

    Pre-training:
    ```bash
    python test_FDA_on_spikes_pretrain.py
    ```

    Fine-tuning:
    ```bash
    python test_FDA_on_spikes_ft.py
    ```


#### Results and Models

We ran `test_FDA_on_spikes_pretrain.py` and `test_FDA_on_spikes_ft.py` on an NVIDIA 1080Ti (11GB) and obtained the $R^2$ values for source and target sessions from the RT-M dataset, as summarized in the table below:

| Day | 1    | 38   | 39   | 40   | 52   | 53   | 67   | 69   | 77   | 79   |
|:---:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| $R^2(\%)$ | 71.84 | 67.73 | 56.05 | 50.79 | 46.20 | 53.46 | 48.05 | 25.85 | 14.64 | 40.21 |

The resulting pre-trained and fine-tuned FDA models on RT-M are publicly available on [Huggingface](https://huggingface.co/plwang/FDA) under the `pre_train` and `ft` folders.


### Citation

```bibtex
@inproceedings{wang2025FDA,
  title     = {Flow Matching for Few-Trial Neural Adaptation with Stable Latent Dynamics},
  author    = {Wang, Puli and Qi, Yu and Wang, Yueming and Pan, Gang},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025}
}
```

### Acknowledgement

We would like to thank the contributors of the [SiT](https://github.com/willisma/SiT), [LDNS](https://github.com/mackelab/LDNS), and [iTransformer](https://github.com/thuml/iTransformer) repositories for making their research openly available.