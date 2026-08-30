from collections import defaultdict
from typing import *
import numpy as np
import torch
from loguru import logger
from torch_geometric.data import Batch as GraphBatch, Data as GraphData
from model.rl import RLModel, BufferEntry
import asyncio
import pandas as pd
import requests
import json
from model.mod_pifold import InverseFoldingDiffusionPiFoldModel
from utils.utils import t2n


async def deterministic_call_esm_kubernetes_endpoint_async(chunk_sequences):
    """
    Calls ESM Kubernetes using synchronous requests
    Returns results directly
    """

    host_ip = "IP GOES HERE"

    # 0 is sequence, 1 is ID
    data = []
    for i in range(len(chunk_sequences)):
        data.append([chunk_sequences[i][1] + f"_{i}", chunk_sequences[i][0]])
    print(f"Starting {[len(i[0]) for i in chunk_sequences]}")
    inf_request = {"seqs": [i[1] for i in data], "uids": [i[0] for i in data]}
    print(inf_request)
    # inf_request = {"columns": ["uid", "seqs"], "data": data}
    successful_request = False
    while not successful_request:
        successful_send = False
        while not successful_send:
            print("Sending request")
            try:
                results = requests.post(
                    f"http://{host_ip}/invocations", json=inf_request, timeout=6000
                )
                print("Successfully sent request")

                dict_data = json.loads(results.json()["result"])
                out_df = pd.DataFrame(
                    data=dict_data["data"], columns=dict_data["columns"]
                )


                successful_send = True
            except Exception as e:
                print(
                    f"Waiting too long got this error {e} so we will resend the request"
                )
        try:
            # out_df = pd.DataFrame(results.json())
            if out_df.size:
                successful_request = True
        except Exception as e:
            successful_request = False
            print(
                f"Got a bad request RETRYING the cluster! Error; {e} and {results.json()}"
            )

    # We want only backbone C, CA, N, O
    out_df = out_df[out_df["atom_name"].isin(["C", "CA", "N", "O"])]
    num_examples = list(set(out_df["uid"].values))
    results = []
    for i in num_examples:
        print(out_df[out_df["uid"] == i].shape)
        results.append(
            [
                torch.Tensor(
                    out_df[out_df["uid"] == i][["x_coord", "y_coord", "z_coord"]]
                    .to_numpy()
                    .reshape((1, -1, 4, 3))
                ),
                i.split("_")[0],
            ]
        )
    return results

class RLStructuralRecoveryModel(RLModel, InverseFoldingDiffusionPiFoldModel):

    def construct(self, config, advantages=None, ngc=None):
        InverseFoldingDiffusionPiFoldModel.construct(self, config)


    async def run_esmfold_server_main(self, sequences, sample_ids):
        """
        Runs ESMFold on server using asyncio python library
        """
        import time
        print(f"Starting for {len(sequences)}")

        initial_start = time.time()
        super_chunk_sequences = []
        chunked_sequences = []
        chunk = []
        release_chunk = False

        for i, z in zip(sequences, sample_ids):
            if len(chunk) == 2 or release_chunk:
                chunked_sequences.append(chunk)
                chunk = []

            if len(i) > 200:
                chunked_sequences.append([[i, z]])
            else:
                chunk.append([i, z])

        # If there is a chunk left then
        if chunk:
            chunked_sequences.append(chunk)


        total_outputs = await asyncio.gather(
            *[
                deterministic_call_esm_kubernetes_endpoint_async(i)
                for i in chunked_sequences
            ]
        )

        print(f"Finished for {len(sequences)} at {time.time() - initial_start}")
        return total_outputs

    def kabsch(self, P, Q):
        """
        Aligns two sets of points P and Q using the Kabsch algorithm.
        P and Q are each numpy arrays of shape (N,3) representing N 3D points.

        The Kabsch algorithm calculates the optimal rotation and translation
        that minimizes the root mean squared deviation (RMSD) between two
        paired sets of points.
        """
        P_ctd, P_adj = self._centroid_adjust(P)
        Q_ctd, Q_adj = self._centroid_adjust(Q)
        h = np.matmul(P_adj.T, Q_adj)
        u, _, vt = np.linalg.svd(h)
        v = vt.T
        d = np.linalg.det(v @ u.T)
        e = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])
        rot = v @ e @ u.T
        tran = Q_ctd - np.matmul(rot, P_ctd)
        return rot, tran

    def _centroid_adjust(self, X):
        X_ctd = np.average(X, axis=0)
        X_adj = X - X_ctd
        return X_ctd, X_adj

    def tm_score_no_align(
        self, A, B, tm_coord_mask=None
    ):  # is_ca: bool = True, coord_mask=None):
        """
        Calculate the *unaligned* TM-score between two sets of points.
        V and W are each numpy arrays of shape (N,3) representing N 3D points.
        d0 is a scaling factor based on the size of the protein.
        tm_coord_mask is a mask to apply to the coordinates before calculating the TM-score
        """
        if tm_coord_mask is not None:
            A = A * tm_coord_mask
            B = B * tm_coord_mask
        # only calculate on C-alpha (as original TM-score paper)
        A = A[1::4, :]
        B = B[1::4, :]
        N = len(A)
        d = np.sqrt(np.sum((A - B) ** 2, axis=1))
        # d0 is less than 0.5 for L < 22
        # and nan for L < 15 (root of a negative number)
        d0 = 1.24 * np.power(N - 15, 1 / 3) - 1.8
        d0 = max(0.5, d0)
        return np.sum(1 / (1 + (d / d0) ** 2)) / N

    def tm_score(self, A, B, A_mask=None, B_mask=None, tm_coord_mask=None):
        """
        Calculate the TM-score between two sets of points.
        V and W are each numpy arrays of shape (N,3) representing N 3D points.
        V_mask and W_mask are the same arrays but masked by plddts scores and solely used for high-confidence alignment
        d0 is a scaling factor based on the size of the protein.
        tm_coord_mask is a mask to apply to the coordinates before calculating the TM-score
        """
        if A_mask is not None and B_mask is None:
            RuntimeError(
                "both structures have to be masked to keep the same number of points"
            )
        if A_mask is None and B_mask is not None:
            RuntimeError(
                "both structures have to be masked to keep the same number of points"
            )
        if A_mask is not None and B_mask is not None:
            rot, tran = self.kabsch(B_mask, A_mask)
        else:
            rot, tran = self.kabsch(B, A)
        B_adj = (rot @ B.T).T + tran
        tm_score = self.tm_score_no_align(
            A, B_adj, tm_coord_mask=tm_coord_mask
        )  # is_ca=is_ca,

        return tm_score

    @torch.no_grad()
    def generate_samples(self, batch: GraphBatch, n_samples: int) -> dict:
        """
        Samples n_samples trajectories for the batch
        """

        buffer = defaultdict(lambda: {t.item(): [] for t in self.timesteps})
        advantages = defaultdict(list)

        keys = [
            "x_0",
            "x_t",
            "x_tm1",
            "x_logprobs",
            "features_0",
            "features_t",
            "features_tm1",
            "features_logprobs",
            "t",
            "batch",
        ]

        def split_batch(G):
            subgraphs = []
            batch_idcs = G.batch
            n_batches = t2n(batch_idcs.max() + 1)

            for ib in range(n_batches):
                mask = batch_idcs == ib
                kw = {}
                for k in G.keys():
                    if k == "name" or k == "names" or k == "ptr" or k == "sample_id":
                        if k == "name" or k == "names":
                            kw["sample_id"] = G[k][ib]
                        else:
                            kw[k] = G[k][ib]
                    elif k == "num_nodes":
                        kw[k] = mask.sum().detach().cpu().clone()
                    elif k == "lengths":
                        kw[k] = torch.tensor([G[k][ib]])
                    elif "pifold" in k:
                        kw[k] = G[k][ib].detach().cpu().clone()
                    else:
                        kw[k] = G[k][mask].detach().cpu().clone()

                # kw["x"] = kw.get("x_t", kw.get("x_0"))  # ???
                # kw["sample_id"] = G.name[ib]

                g = GraphData(**kw)
                # g.num_nodes = G["x_0"][mask].shape[0]
                subgraphs.append(g)

            return subgraphs

        batch["features_true"] = batch["features_0"]
        num = batch["mask_pifold"].shape[0]
        subgraphs = split_batch(batch)
        tiled_batch = GraphBatch.from_data_list(subgraphs * n_samples)
        batch = self.init_sample_batch(tiled_batch)
        for i in batch.keys():
            if "pifold" in i:
                batch[i] = batch[i].view(num * 4, -1, *batch[i].shape[1:])

        for t in self.timesteps:
            t = t.item()

            # each step of this loop takes in x_t and estimates x_t-1
            if t != self.T:
                batch.features_t = batch.features_step.clone()

            batch = self.denoise_batch_with_logprobs(batch, t)

            # store the current step
            split = split_batch(batch)
            # assert hasattr(GraphBatch.from_data_list(split), "batch")
            for g in split:
                buffer[g.sample_id][t].append(g)

        # ft_scale = self.config.feature_scale if self.config.scale_features else 1.0
        ft_scale = 1

        def get_sequences():
            sequences = []
            res_sample_id = []
            real_structure = []
            true_sequences = []

            last_t = self.timesteps[-1].item()

            for sample_id, traj in buffer.items():
                for g in traj[last_t]:
                    # t=1 => tm1=0, which is the one we want to score
                    # Instead of true features get the true structure
                    # ft = g["features_true"]
                    ft = g["x"]
                    fp = g["features_t"]
                    fp = fp.argmax(axis=-1)
                    mask = g["mask"].bool()

                    ft, fp = ft[mask], fp[mask]
                    alphabet = "ACDEFGHIKLMNPQRSTVWY"
                    index_to_AA = {i: a for i, a in enumerate(alphabet)}
                    pred_sequence = "".join(np.vectorize(index_to_AA.get)(fp).tolist())
                    true_sequence = "".join(
                        np.vectorize(index_to_AA.get)(
                            g["features_true"].argmax(axis=-1)
                        ).tolist()
                    )
                    sequences.append(pred_sequence)
                    res_sample_id.append(sample_id)
                    real_structure.append(ft)
                    true_sequences.append(true_sequence)

            return sequences, res_sample_id, real_structure, true_sequences

        # compute and assign advantages
        async def get_advantages():
            tasks = []
            last_t = self.timesteps[-1].item()
            for sample_id, traj in buffer.items():
                for g in traj[last_t]:
                    # t=1 => tm1=0, which is the one we want to score
                    # Instead of true features get the true structure
                    # ft = g["features_true"]
                    ft = g["x"]
                    fp = g["features_t"]
                    fp = fp.argmax(axis=-1)
                    mask = g["mask"].bool()

                    ft, fp = ft[mask], fp[mask]
                    alphabet = "ACDEFGHIKLMNPQRSTVWY"
                    index_to_AA = {i: a for i, a in enumerate(alphabet)}
                    pred_sequence = "".join(np.vectorize(index_to_AA.get)(fp).tolist())

                    tasks.append(
                        self.advantage_fn(
                            sample_id=sample_id,
                            real_structure=ft,
                            pred_sequence=pred_sequence,
                        )
                    )
            return await asyncio.gather(*tasks)

        predicted_sequences, res_sample_ids, real_structure_compare, true_sequences = (
            get_sequences()
        )

        si_to_seq = {}
        seq_to_si = {}
        ids_scanned = []
        real_seqs_fold = []
        for i, z in zip(res_sample_ids, true_sequences):
            if i not in si_to_seq:
                si_to_seq[i] = z
                seq_to_si[z] = i
                ids_scanned.append(i)
                real_seqs_fold.append(z)

        true_structure_compare = asyncio.run(
            self.run_esmfold_server_main(real_seqs_fold, ids_scanned)
        )

        si_to_reference = {}
        for x in true_structure_compare:
            for z, i in x:
                si_to_reference[i] = z.squeeze(0).view(-1, 3).numpy()

        predicted_structure_compare = asyncio.run(
            self.run_esmfold_server_main(predicted_sequences, res_sample_ids)
        )

        for x in predicted_structure_compare:
            for z, i in x:
                pred_struct = z.squeeze(0).view(-1, 3).numpy()
                true_struct = si_to_reference[i]
                advantages[i].append([self.tm_score(true_struct, pred_struct)])

        return dict(trajectories=dict(buffer), advantages=dict(advantages))

    def collate_fn_buffer(self, batch: List[BufferEntry]) -> GraphBatch:
        return GraphBatch.from_data_list([b.sample for b in batch])

    def ppo_step(self, batch: List[BufferEntry], mode, eps_clip: float):
        model_batch = self.colocate_data(self.collate_fn_buffer(batch))

        if mode == 0:
            # self.train()
            self.eval()
            logprobs_new = self.compute_logprobs(model_batch)
        else:
            self.eval()
            logprobs_new = self.compute_logprobs_nograd(model_batch)

        logprobs_old = []
        for b in batch:
            s = b.sample
            sub_logprob = s.features_logprobs
            logprobs_old.append(sub_logprob.sum())

        logprobs_old = torch.tensor(logprobs_old, device=logprobs_new.device)

        # logprobs_old = torch.tensor([s.x_logprobs + s.features_logprobs for s in batch], device=logprobs_new.device)
        # Compute PPO objective
        ratio = torch.exp(logprobs_new - logprobs_old)

        if torch.isnan(ratio).any():
            logger.info("Hit a NaN!")
            breakpoint()

        print(f"Ratio: {ratio}", flush=True)

        adv = self.colocate_data(torch.tensor([s.advantage for s in batch]))
        og_adv = self.colocate_data(torch.tensor([s.original_advantage for s in batch]))
        print(f"Advantage: {adv}", flush=True)
        surr1 = ratio * adv
        # print(f"Surr1: {surr1}")
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv
        # print(f"Surr2: {surr2}")
        loss = -torch.min(surr1, surr2)
        print(f"Loss: {loss}", flush=True)
        clip_count = 1 / loss.shape[0] * (loss != -surr1).sum()
        print(f"Clip count: {clip_count}", flush=True)

        # return
        if mode == 0:
            return {
                "loss": loss.mean(),
                "advantage": adv.mean(),
                "advantage_std": adv.std(),
                "original_advantage": og_adv.mean(),
                "original_advantage_std": og_adv.std(),
                "ratio": ratio.mean(),
                "clip_count": clip_count,
            }
        elif mode == 1:
            return {
                "loss": loss.mean(),
                "advantage": adv.mean(),
                "advantage_std": adv.std(),
                "original_advantage": og_adv.mean(),
                "original_advantage_std": og_adv.std(),
                "ratio": ratio.mean(),
                "clip_count": clip_count,
            }

    def compute_logprobs(self, batch: GraphBatch) -> GraphBatch:
        output = self(batch, logprobs=False)
        t_l = batch.t.long()
        logits = self.p_pred(t_l, batch.features_t, output.features_pred)
        sample = batch.features_step
        features_logprobs = self.log_sample_categorical_with_logprobs(
            sample, logits.log()
        )
        result = []
        for i in range(batch["batch"].max() + 1):
            mask = batch["batch"] == i
            result.append(features_logprobs[mask].sum())

        return torch.stack(result)