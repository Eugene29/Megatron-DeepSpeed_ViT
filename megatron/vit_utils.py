# from torch.profiler import profile, record_function, ProfilerActivity, schedule
# import numpy as np
# import pytz
import subprocess as sp
# from threading import Thread , Timer
# import sched, time
# import torch
# from training import pretrain

def get_gpu_utilization():
    # Run nvidia-smi command with query options
    result = sp.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        stdout=sp.PIPE,
        encoding="utf-8"
    )
    # Process output into a list of GPU utilization values
    utilization = [int(x) for x in result.stdout.strip().split("\n")]
    return utilization

def get_gpu_memory():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    #ACCEPTABLE_AVAILABLE_MEMORY = 1024*1024
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
    # print(memory_use_values)
    return (memory_use_values)

if __name__ == "__main__":
    # from mpi4py import MPI
    # COMM = MPI.COMM_WORLD
    # RANK = COMM.Get_rank()
    # WS = COMM.Get_size()
    import os
    RANK = int(os.environ["RANK"])
    WS = int(os.environ["WORLD_SIZE"])

    import torch
    torch.set_default_device(RANK)

    tensor = torch.randn(1000, 1000, 1000)
    
    print(f"get_gpu_memory(): {get_gpu_memory()}")
    # print(f"get_gpu_utilization(): {get_gpu_utilization()}")

    ## torchrun --nproc-per-node 4 /eagle/datascience/eku/Megatron-DeepSpeed_ViT/megatron/vit_utils.py