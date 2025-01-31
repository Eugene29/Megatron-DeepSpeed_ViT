## Welcome to ALCF ViT repo

## Clone & Init Submodule:
```
  git clone --recursive https://github.com/Eugene29/Megatron-DeepSpeed_ViT.git ## Clone module + submodule
  cd Megatron-DeepSpeed_ViT
  git submodule update --init --recursive ## Init & Update submodule
```

## Environment
Only base environment is needed for polaris cluster while for aurora, we employ sam's ezpz library. A suitable virtual environment (in flare file-system) is activated by automatically on aurora. 

## Notes:
Main script for entry is mult_mds.sh. In here, you'll need to modify SCRIPT_DIR. There is also descriptions for possible flags. Any other fixed ENV and MDS-related variables can be changed in mult_launch.sh. 