
# export NUM_EPOCHS=500 
qsub -V -A datascience -q debug -l select=1 -l walltime=1:00:00,filesystems=eagle:home /eagle/datascience/eku/Megatron-DeepSpeed_ViT/mult_mds.sh
# export SP=4
# qsub -V -A datascience -q debug -l select=1 -l walltime=1:00:00,filesystems=eagle:home /eagle/datascience/eku/Megatron-DeepSpeed_ViT/mult_mds.sh
