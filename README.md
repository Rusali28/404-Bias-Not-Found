# PAAC Extensions for Popularity Bias Mitigation

## Overview
We extend the PAAC (Popularity-Aware Alignment and Contrast) framework to better mitigate popularity bias in recommendation systems. PAAC uses contrastive learning with separate objectives for popular and unpopular items. In this project, we introduce contrastive temperature scaling based on item popularity and dynamic gradient reweighting to improve generalization. We also find that a 55-45 split between popular and unpopular items yields the best NDCG results on Yelp and Gowalla. Our enhancements boost PAAC’s robustness and fairness in recommendation quality.

## Key Contributions
- Meta Learning & Gradient Weight Learning
- Popularity Ratio Sensitivity
- Adaptive Temperature Scaling
- Extending Domain Beyond E-Commerce

## Datasets
- Yelp
- Gowalla
- MOOC

## Acknowledgements
The original research paper:
[Popularity-Aware Alignment and Contrast for Mitigating Popularity Bias](https://arxiv.org/pdf/2405.20718)

## Requirements

- Python == 3.9.7
- PyTorch == 1.12.0+cu113
- Numba == 0.54.1
- NumPy == 1.20.0
- FAISS-GPU == 1.7.2
- Pandas == 1.3.4

## Branches Information:

- main branch implements Tempertaure Scaling
- popularity-ratio-sensitivty implements Popularity Ratio Sensitivity
- meta-learning implements Gradient Reweighting and Meta Learning
- mooc implements the Extension eyon E-Commerce

For models using LightGCN as the backbone, you can run the following commands:

### Gowalla

```bash
python PAAC_main.py --dataset_name gowalla --layers_list '[6]' --cl_rate_list '[5]' --align_reg_list '[50]' --lambada_list '[0.2]' --gama_list '[0.2]'
```

### Yelp2018

```bash
python PAAC_main.py --dataset_name yelp2018 --layers_list '[5]' --cl_rate_list '[10]' --align_reg_list '[1e3]' --lambada_list '[0.8]' --gama_list '[0.8]'
```

### MOOC
```bash
python PAAC_main.py --dataset_name mooc --layers_list '[5]' --cl_rate_list '[10]' --align_reg_list '[1e3]' --lambada_list '[0.8]' --gama_list '[0.8]'
```

## Documentation

[Report](https://arxiv.org/pdf/2405.20718)





 
