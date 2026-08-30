import time
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections.abc import Mapping, Sequence

# Code from PiFold!, utility code they used for their featurization

# Thanks for StructTrans
# https://github.com/jingraham/neurips19-graph-protein-design
def nan_to_num(tensor, nan=0.0):
    idx = torch.isnan(tensor)
    tensor[idx] = nan
    return tensor


def _normalize(tensor, dim=-1):
    return nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def cal_dihedral(X, eps=1e-7):
    dX = X[:, 1:, :] - X[:, :-1, :]  # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0 = U[:, :-2, :]  # CA-N, C-CA, N-C,...
    u_1 = U[
        :, 1:-1, :
    ]  # C-CA, N-C, CA-N, ... 0, psi_{i}, omega_{i}, phi_{i+1} or 0, tau_{i},...
    u_2 = U[:, 2:, :]  # N-C, CA-N, C-CA, ...

    n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_2), dim=-1)

    cosD = (n_0 * n_1).sum(-1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)

    v = _normalize(torch.cross(n_0, n_1), dim=-1)
    D = torch.sign((-v * u_1).sum(-1)) * torch.acos(cosD)  # TODO: sign

    return D


def _dihedrals(X, dihedral_type=0, eps=1e-7):
    B, N, _, _ = X.shape
    # psi, omega, phi
    X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)  # ['N', 'CA', 'C', 'O']
    D = cal_dihedral(X)
    D = F.pad(D, (1, 2), "constant", 0)
    D = D.view((D.size(0), int(D.size(1) / 3), 3))
    Dihedral_Angle_features = torch.cat((torch.cos(D), torch.sin(D)), 2)

    # alpha, beta, gamma
    dX = X[:, 1:, :] - X[:, :-1, :]  # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0 = U[:, :-2, :]  # CA-N, C-CA, N-C,...
    u_1 = U[:, 1:-1, :]  # C-CA, N-C, CA-N, ...
    cosD = (u_0 * u_1).sum(-1)  # alpha_{i}, gamma_{i}, beta_{i+1}
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.acos(cosD)
    D = F.pad(D, (1, 2), "constant", 0)
    D = D.view((D.size(0), int(D.size(1) / 3), 3))
    Angle_features = torch.cat((torch.cos(D), torch.sin(D)), 2)

    D_features = torch.cat((Dihedral_Angle_features, Angle_features), 2)
    return D_features


def _rbf(D, num_rbf):
    D_min, D_max, D_count = 0.0, 20.0, num_rbf
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu.view([1, 1, 1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


def _get_rbf(A, B, E_idx=None, num_rbf=16):
    if E_idx is not None:
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = _rbf(D_A_B_neighbors, num_rbf)
    else:
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, :, None, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        RBF_A_B = _rbf(D_A_B, num_rbf)
    return RBF_A_B


def _orientations_coarse_gl_tuple(X, E_idx, eps=1e-6):
    V = X.clone()
    X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)
    dX = X[:, 1:, :] - X[:, :-1, :]  # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0, u_1 = U[:, :-2, :], U[:, 1:-1, :]
    n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
    b_1 = _normalize(u_0 - u_1, dim=-1)

    n_0 = n_0[:, ::3, :]
    b_1 = b_1[:, ::3, :]
    X = X[:, ::3, :]
    Q = torch.stack((b_1, n_0, torch.cross(b_1, n_0)), 2)
    Q = Q.view(list(Q.shape[:2]) + [9])
    Q = F.pad(Q, (0, 0, 0, 1), "constant", 0)  # [16, 464, 9]

    Q_neighbors = gather_nodes(Q, E_idx)  # [16, 464, 30, 9]
    X_neighbors = gather_nodes(V[:, :, 1, :], E_idx)  # [16, 464, 30, 3]
    N_neighbors = gather_nodes(V[:, :, 0, :], E_idx)
    C_neighbors = gather_nodes(V[:, :, 2, :], E_idx)
    O_neighbors = gather_nodes(V[:, :, 3, :], E_idx)

    Q = Q.view(list(Q.shape[:2]) + [3, 3]).unsqueeze(2)  # [16, 464, 1, 3, 3]
    Q_neighbors = Q_neighbors.view(
        list(Q_neighbors.shape[:3]) + [3, 3]
    )  # [16, 464, 30, 3, 3]

    dX = (
        torch.stack([X_neighbors, N_neighbors, C_neighbors, O_neighbors], dim=3)
        - X[:, :, None, None, :]
    ) 
    dU = torch.matmul(Q[:, :, :, None, :, :], dX[..., None]).squeeze(
        -1
    )  
    B, N, K = dU.shape[:3]
    E_direct = _normalize(dU, dim=-1)
    E_direct = E_direct.reshape(B, N, K, -1)
    R = torch.matmul(Q.transpose(-1, -2), Q_neighbors)
    q = _quaternions(R)


    dX_inner = V[:, :, [0, 2, 3], :] - X.unsqueeze(-2)
    dU_inner = torch.matmul(Q, dX_inner.unsqueeze(-1)).squeeze(-1)
    dU_inner = _normalize(dU_inner, dim=-1)
    V_direct = dU_inner.reshape(B, N, -1)
    return V_direct, E_direct, q


def gather_edges(edges, neighbor_idx):
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)


def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view(
        (neighbor_idx.shape[0], -1)
    )  # [4, 317, 30]-->[4, 9510]
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(
        -1, -1, nodes.size(2)
    )  # [4, 9510, dim]
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)  # [4, 9510, dim]
    return neighbor_features.view(
        list(neighbor_idx.shape)[:3] + [-1]
    )  # [4, 317, 30, 128]


def _quaternions(R):
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(
        torch.abs(
            1 + torch.stack([Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz], -1)
        )
    )
    _R = lambda i, j: R[:, :, :, i, j]
    signs = torch.sign(
        torch.stack([_R(2, 1) - _R(1, 2), _R(0, 2) - _R(2, 0), _R(1, 0) - _R(0, 1)], -1)
    )
    xyz = signs * magnitudes
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.0
    Q = torch.cat((xyz, w), -1)
    return _normalize(Q, dim=-1)


def cuda(obj, *args, **kwargs):
    """
    Transfer any nested container of tensors to CUDA.
    """
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, Mapping):
        return type(obj)({k: cuda(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, Sequence):
        return type(obj)(cuda(x, *args, **kwargs) for x in obj)
    elif isinstance(obj, np.ndarray):
        return torch.tensor(obj, *args, **kwargs)

    raise TypeError("Can't transfer object type `%s`" % type(obj))


"""============================================================================================="""
""" Graph Encoder """
"""============================================================================================="""


def get_attend_mask(idx, mask):
    mask_attend = gather_nodes(mask.unsqueeze(-1), idx).squeeze(
        -1
    )  
    mask_attend = mask.unsqueeze(-1) * mask_attend  
    return mask_attend


def _full_dist(X, mask, top_k=30, eps=1e-6):
    # mask of C-alpha atoms, don't want to do dist on C-alphas that don't exist
    mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
    dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
    D = (1.0 - mask_2D) * 10000 + mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

    D_max, _ = torch.max(D, -1, keepdim=True)
    D_adjust = D + (1.0 - mask_2D) * (D_max + 1)
    D_neighbors, E_idx = torch.topk(
        D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False
    )
    return D_neighbors, E_idx
