## Welcome to ALCF ViT repo, maintained by Eugene Ku

## Clone & Init Submodule:
```
  git clone --recursive https://github.com/Eugene29/Megatron-DeepSpeed_ViT.git ## Clone module + submodule
  cd Megatron-DeepSpeed_ViT
  git submodule update --init --recursive ## Init & Update submodule
```

## Environment
  Base (Polaris) environment + DeepSpeed(15.1 or higher(?)) + ezpz (sam's library)

- ### Installation
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
- ### Modify `mds_args.sh`
  - Change env activation in `mds_args.sh`

## Functionality
  To know about all available functionalities and how to use them, checkout the documentation in `mult_mds.sh`
  
## Script
  For consecutive runs one can:
  ```
  bash mult_mds.sh
  ```
  
  For submitting multiple jobs:
  ```
  bash mult_qsub.sh
  ```

## Note:
  - Most comments written by me are flagged with ## instead of #
