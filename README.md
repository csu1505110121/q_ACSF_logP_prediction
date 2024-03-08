# q_ACSF_logP_prediction

This project is for accurate and efficient prediction of partition coefficient ($\log P$) encoding both entropy and polarization effects into descriptors. Armed with high-dimensional neural networks, satisfied results are achieved compared with state-of-art methods. 

## Architecture
```
- logp_prediction.yml: env for logp training and prediction and visualization, such as tensorflow, matplotlib, pandas, numpy, and so on.
- md_openmm.yml: env for MD simulations

- logp/DATASETS: dir stores datasets utilized in this project, consists of SMILES and their corresponding logp values measured experimentally
- logp/io/elem_filter.py: module for filtering molecules of only element C H O N;
- logp/sf_behler/dump.py: module for dumping coord into xyz or pdb format file;
- logp/sf_behler/g3D.py: module for convert SMILEs into 3D structures using RDKit package;
- logp/sf_behler/sf.py: module for calculating ACSFs, different weights,such as charge, ensemble weight could be parsed into this function;
- logp/sf_behler/utils.py: util tools for building high-dimensional neural network;
- logp/sf_behler/G_sf_generator.py: wrapup ACSFs related functions

```







For more details, you could consult this [paper](https://pubs.rsc.org/en/content/articlelanding/2022/CP/D2CP02648A)
```
@article{zhu2022molecular,
  title={Molecular partition coefficient from machine learning with polarization and entropy embedded atom-centered symmetry functions},
  author={Zhu, Qiang and Jia, Qingqing and Liu, Ziteng and Ge, Yang and Gu, Xu and Cui, Ziyi and Fan, Mengting and Ma, Jing},
  journal={Physical Chemistry Chemical Physics},
  volume={24},
  number={38},
  pages={23082--23088},
  year={2022},
  publisher={Royal Society of Chemistry}
}
```

