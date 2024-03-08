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
- logp/zq/feat/feat_score.py: estimating importance of conventional descriptors derived from RDKit with methods, such as, SelectKBest, f_regression, RFE
- logp/zq/io/base.py: provide basic function, such as sparse function;
- logp/zq/io/load_dataframe.py: load dataframe results from files;
- logp/zq/io/load_desc.py: load descriptors generated;
- logp/zq/io/log2feat.py: convert log file generated by Gaussian into features;
- logp/zq/io/pdb2feat.py: convert pdb file into features;
- logp/zq/io/smi2feat.py: convert SMILEs into features;

- logp/zq/model/model.py: module for constructing neural network model;
- logp/zq/network/network.py: implementation of high-dimensional neural network (HDNN);

- logp/zq/utils/md/boltz.py: calc prob weight according to Boltzmann;
- logp/zq/utils/md/load_prop.py: load props from intermediate files;
- logp/zq/utils/md/md.py: module for performing vacuum md using OpenMM;
- logp/zq/utils/md/md_analysis.py: module for MD analysis, such as, RMSD;


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

