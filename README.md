## Welcome to ALCF ViT repo

## Clone & Init Submodule:
```
  git clone --recursive https://github.com/Eugene29/Megatron-DeepSpeed_ViT.git ## Clone module + submodule
  cd Megatron-DeepSpeed_ViT
  git submodule update --init --recursive ## Init & Update submodule
```

## Environment
Only base environment is needed for polaris cluster and for aurora, we use sam's ezpz repo. A suitable virtual environment is activated by default on aurora. 

## Notes:
Main script for entry is mult_mds.sh. In here, you'll need to modify SCRIPT_DIR. There is also argument descriptions provided. 

Two set-up scripts are mds_args.sh and mds_launch.sh. Let me know if you think they will be better off merged. 
