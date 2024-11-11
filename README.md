## Welcome to ALCF ViT repo, maintained by Eugene Ku

## Environment
Base (Polaris) environment + DeepSpeed(15.1 or higher?) + ezpz (sam's library)

### Installation
1. ezpz:
```
git clone https://github.com/saforem2/ezpz;
cd ezpz;
pip install .
```

2. DeepSpeed
```
pip install deepspeed==0.15.1
```

Change env activation in `mds_args.sh`

## Functionality
To check available functionalities and how to use them, checkout the documentation in `mult_mds.sh`
## Script
For consecutive runs one can:
```
bash mult_mds.sh
```

For submitting multiple jobs:
```
bash mult_qsub.sh
```
