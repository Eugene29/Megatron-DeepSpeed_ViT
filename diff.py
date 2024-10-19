import torch
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--MODE1", type=str, required=True)
# parser.add_argument("--MODE2", type=str, required=True)
# parser.add_argument("--DS_DIR", type=str, default="")
# parser.add_argument("--FA", type=str, default="")
# args = parser.parse_args()

# TEMP_DS - FA
# if "DeepSpeed" in args.DS_DIR:
#     DS_PREFIX="ds15.2"
# elif "Test" in args.DS_DIR:
#     DS_PREFIX="ds14.2"
# else:
#     DS_PREFIX="ds14.4"
# if args.FA == "1":
#     FA_PREFIX = "FA_"
# else:
#     FA_PREFIX = ""

# save_dir=f"{FA_PREFIX}{DS_PREFIX}/"

DP_DIR = '/eagle/datascience/eku/Megatron-DeepSpeed_ViT/debug/output_DP.txt.pt'
SP_DIR = '/eagle/datascience/eku/Megatron-DeepSpeed_ViT/debug/output_SP.txt.pt'

grad_bef_weights_aft_weights_DP = torch.load(DP_DIR)
grad_bef_weights_aft_weights_SP = torch.load(SP_DIR)
gradients_DP, bef_weights_DP, aft_weights_DP = grad_bef_weights_aft_weights_DP
gradients_SP, bef_weights_SP, aft_weights_SP = grad_bef_weights_aft_weights_SP

dev = torch.device("cuda")
diff_grad_abs_norm = 0
for grad1, grad2 in zip(gradients_DP, gradients_SP):
    grad1 = grad1.to(dev)
    grad2 = grad2.to(dev)
    grad1[grad1.isinf()] = 0
    grad2[grad2.isinf()] = 0

    # if torch.all(grad1.isnan().sum() > 0) or torch.all(grad2.isnan().sum()) > 0:
    #     print(f"grad1: {grad1}")
    #     print(f"grad2: {grad2}")

    abs_diff_norm = (grad1.to(dev) - grad2.to(dev)).abs().sum()
    diff_grad_abs_norm += abs_diff_norm
    ## ISSUE: One of them is infinity (which doesn't makes sense given that the max difference is 16.)
    # print(f"diff_grad_abs_norm: {diff_grad_abs_norm}")
    # if abs_diff_norm == float('inf'):
    #     # torch.set_printoptions(threshold=torch.inf)
    #     # print(grad1.to(dev) - grad2.to(dev))
    #     print(f"grad1.max(): {grad1.max()}")
    #     print(f"grad2.max(): {grad2.max()}")
    #     print(f"(grad1 - grad2).max(): {(grad1 - grad2).max()}")
    #     diffs = grad1 - grad2
    #     manual = 0
    #     for diff in diffs:
    #         manual += diff.abs()
    # raise KeyboardInterrupt

diff_bef_weights_abs_norm = 0
for bef1, bef2 in zip(bef_weights_DP, bef_weights_SP):
    diff_bef_weights_abs_norm += (bef1.to(dev) - bef2.to(dev)).abs().sum()

diff_aft_weights_abs_norm = 0
for aft1, aft2 in zip(aft_weights_DP, aft_weights_SP):
    diff_aft_weights_abs_norm += (aft1.to(dev) - aft2.to(dev)).abs().sum()


# print(f"loss1: {loss1}")
# print(f"loss2: {loss2}")
# print(f"loss difference: {(loss1 - loss2).abs()}")
# print(f"diff_grad_abs_norm: {diff_grad_abs_norm}")

with open("/eagle/datascience/eku/Megatron-DeepSpeed_ViT/debug/diff.log", "w") as f:
    f.write(f"\ndiff_grad_abs_norm: {diff_grad_abs_norm}")
    f.write(f"\ndiff_bef_weights_abs_norm: {diff_bef_weights_abs_norm}")
    f.write(f"\ndiff_aft_weights_abs_norm: {diff_aft_weights_abs_norm}")

    # f.write(f"diff_grad_abs_norm: {diff_grad_abs_norm}")
    # f.write(f"diff_bef_weights_abs_norm: {diff_bef_weights_abs_norm}")
    # f.write(f"diff_aft_weights_abs_norm: {diff_aft_weights_abs_norm}")

    # f.write(f"loss1: {loss1}")
    # f.write(f"loss2: {loss2}")
    # f.write(f"loss difference: {(loss1 - loss2).abs()}")
    # f.write(f"diff_grad_abs_norm: {diff_grad_abs_norm}")