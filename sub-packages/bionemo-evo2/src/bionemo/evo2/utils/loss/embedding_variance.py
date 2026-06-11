# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This assumes you have a parallel_state module similar to Megatron-LM's.
# If not, you'll need to pass tp_world_size and tp_group directly if they
# are obtained differently.
# from megatron.core import parallel_state # Example import


import torch
import torch.distributed
from megatron.core import parallel_state
from torch.autograd import Function


class SquaredErrorTargetedVarianceLossFunction(Function):
    """This loss function is used to calculate the loss based on the squared difference between the global mean of per-word variances and target."""

    @staticmethod
    def forward(ctx, we_weight: torch.Tensor, loss_coeff: float, var_target: float) -> torch.Tensor:
        """Calculates a loss based on the squared difference between the global mean of per-word variances and target.

        Assumes vocab-parallel sharding for we_weight (dim 0 is sharded).

        Args:
            ctx (torch.autograd.FunctionContext): Context object for backward pass.
            we_weight (torch.Tensor): Local shard of embedding weights (V_local, H).
            loss_coeff (float): Loss coefficient.
            var_target (float): Targeted variance for the embedding weights.

        Returns:
            torch.Tensor: Scalar loss value.

            weights
        """
        if not we_weight.is_floating_point():
            we_weight = we_weight.float()

        V_local, H = we_weight.shape  # V_local: words on this rank, H: embedding dim

        # Save dimensions for backward pass
        ctx.H_embedding_dim = H
        ctx.V_local_word_count = V_local
        ctx.loss_coeff = loss_coeff
        ctx.var_target = var_target

        # Handle H=0 edge case (embedding dimension is zero)
        if H == 0:
            ctx.is_H_dim_zero = True
            # Mean variance is 0 if H=0. Loss is based on (0 - VAR_TARGET)^2.
            loss_value = loss_coeff * (0.0 - var_target) ** 2
            final_loss_tensor = torch.tensor(loss_value, device=we_weight.device, dtype=we_weight.dtype)
            # Save we_weight for shape, None for we_mean_per_word and V_final (as they are not well-defined or zero)
            ctx.save_for_backward(we_weight, None, None)
            return final_loss_tensor
        ctx.is_H_dim_zero = False

        # Get TP info (assuming parallel_state is globally accessible)
        # Ensure parallel_state is imported and available in the execution scope.
        # from some_module import parallel_state # Make sure this is accessible
        tp_world_size = parallel_state.get_tensor_model_parallel_world_size() or 1
        tp_group = parallel_state.get_tensor_model_parallel_group()  # Can be None
        ctx.tp_world_size_val = tp_world_size

        # 1. Per-word mean (across embedding dimension H)
        # Shape: (V_local, 1)
        we_mean_per_word = we_weight.mean(dim=1, keepdim=True)

        # 2. Per-word variance (across embedding dimension H)
        # we_sq_diffs_per_word shape: (V_local, H)
        we_sq_diffs_per_word = (we_weight - we_mean_per_word) ** 2
        # we_var_per_word_local shape: (V_local,) (biased variance)
        we_var_per_word_local = we_sq_diffs_per_word.mean(dim=1, keepdim=False)

        # 3. Mean of these per-word variances *on this local rank*
        # v_local_mean_of_vars shape: scalar tensor
        v_local_mean_of_vars = torch.tensor(0.0, device=we_weight.device, dtype=we_weight.dtype)
        if V_local > 0:  # Avoid NaN from mean of empty tensor if V_local is 0
            v_local_mean_of_vars = we_var_per_word_local.mean(dim=0, keepdim=False)

        # 4. Globally average these local mean variances
        # V_final_globally_avg_var is the V in the loss formula L = alpha*(V-T)^2
        V_final_globally_avg_var = v_local_mean_of_vars.clone()
        if tp_world_size > 1:
            # Computes V_final = (1/tp_world_size) * sum(v_local_mean_of_vars from each rank)
            V_final_globally_avg_var /= tp_world_size
            torch.distributed.all_reduce(V_final_globally_avg_var, group=tp_group, op=torch.distributed.ReduceOp.SUM)

        # 5. Calculate final loss: LOSS_COEFF * (V_final - VAR_TARGET)^2
        final_loss = loss_coeff * (V_final_globally_avg_var - var_target) ** 2

        # Save tensors needed for gradient computation in backward
        ctx.save_for_backward(we_weight, we_mean_per_word, V_final_globally_avg_var)
        # Other necessary scalars (H, V_local, tp_world_size) are already on ctx.

        return final_loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Backward pass for the SquaredErrorTargetedVarianceLossFunction."""
        we_weight, we_mean_per_word, V_final_saved = ctx.saved_tensors

        # Handle H=0 edge case (gradient is zero)
        if getattr(ctx, "is_H_dim_zero", False):
            return torch.zeros_like(we_weight), None, None  # Grad for we_weight only

        H = ctx.H_embedding_dim
        V_local = ctx.V_local_word_count
        tp_world_size = ctx.tp_world_size_val
        loss_coeff = ctx.loss_coeff
        var_target = ctx.var_target

        # Handle V_local=0 edge case (no words on this rank, so no gradient)
        if V_local == 0:
            return torch.zeros_like(we_weight), None, None  # Grad for we_weight only

        # Chain rule: d(TotalLoss)/dw = d(TotalLoss)/d(final_loss) * d(final_loss)/dw
        # grad_output is d(TotalLoss)/d(final_loss)

        # 1. Calculate d(final_loss) / d(V_final_saved)
        # final_loss = LOSS_COEFF * (V_final_saved - VAR_TARGET)**2
        # dL_dV_final is d(final_loss) / d(V_final_saved)
        dL_dV_final = loss_coeff * 2.0 * (V_final_saved - var_target)

        # grad_V_final is d(TotalLoss) / d(V_final_saved)
        grad_V_final = grad_output * dL_dV_final  # Scalar

        # 2. Propagate gradient from V_final_saved to v_local_mean_of_vars (on current rank)
        # V_final_saved = (1/tp_world_size) * sum_k(v_local_mean_of_vars_k)
        # So, d(V_final_saved) / d(v_local_mean_of_vars_current_rank) = 1 / tp_world_size
        # grad_v_local_mean is d(TotalLoss) / d(v_local_mean_of_vars_current_rank)
        grad_v_local_mean = grad_V_final * (1.0 / tp_world_size)  # Scalar

        # 3. Propagate gradient from v_local_mean_of_vars to we_var_per_word_local_i
        # v_local_mean_of_vars = mean(we_var_per_word_local) = (1/V_local) * sum_i(we_var_per_word_local_i)
        # So, d(v_local_mean_of_vars) / d(we_var_per_word_local_i) = 1 / V_local
        # The coefficient to apply for the next step of chain rule:
        # This is grad_v_local_mean scaled by (1/V_local)
        # This represents d(TotalLoss)/d(we_var_per_word_local_i), assuming it's uniform.
        coeff_for_per_word_var_grad = grad_v_local_mean * (1.0 / V_local)  # Scalar

        # 4. Propagate gradient from we_var_per_word_local_i to we_weight_ik
        # we_var_per_word_local_i = (1/H) * sum_k (we_weight_ik - we_mean_per_word_i[0])^2
        # d(we_var_per_word_local_i) / d(we_weight_ik) = (2/H) * (we_weight_ik - we_mean_per_word_i[0])
        # The term (we_weight_ik - we_mean_per_word_i[0]) is (we_weight - we_mean_per_word)

        # Combine coefficients for the (we_weight - we_mean_per_word) term:
        # This is coeff_for_per_word_var_grad * (2/H)
        final_scalar_coefficient = coeff_for_per_word_var_grad * (2.0 / H)

        grad_we_weight = final_scalar_coefficient * (we_weight - we_mean_per_word)

        # The forward function only takes we_weight as a tensor input requiring grad, the other two inputs
        # are floats and do not get gradients.
        return grad_we_weight, None, None


class SquaredErrorTargetedVarianceLoss(torch.nn.Module):
    """Applies a loss that will encourage variance of some parameter to be close to var_target."""

    def __init__(self, loss_coeff: float = 0.1, var_target: float = 1.0):
        """Applies a loss that will encourage variance of some parameter to be close to var_target.

        Args:
            loss_coeff: Loss coefficient. Defaults to 0.1.
            var_target: targetted variance for the embedding weights. Defaults to 1.0.
        """
        super().__init__()
        self.loss_coeff = loss_coeff
        self.var_target = var_target

    def forward(self, we_weight: torch.Tensor) -> torch.Tensor:
        """Applies the loss to the embedding weights with the user requested loss coefficient and targeted variance.

        Args:
            we_weight: Embedding weights.

        Returns:
            torch.Tensor: Loss value.
        """
        return SquaredErrorTargetedVarianceLossFunction.apply(we_weight, self.loss_coeff, self.var_target)
