# Data

The NASA C-MAPSS turbofan degradation data are not redistributed here. Download them from the
NASA Prognostics Center of Excellence data repository ("Turbofan Engine Degradation Simulation",
Saxena et al., 2008) and place the files so that this path exists:

    CMAPSSData/train_FD004.txt
    CMAPSSData/train_FD002.txt   (FD002 replication, notebooks 13–15)

Only these two training files are used. Point the notebooks at the folder that contains `CMAPSSData`
by setting the environment variable `FD004_ROOT` to the repository root.

`artifacts/FD004_baseline_tensors.npz` (the IID windows produced by notebook 01) is included
(3.8 MB) because the IID experiments start from it.
