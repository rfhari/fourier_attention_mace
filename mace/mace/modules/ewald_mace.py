import torch
import torch.nn as nn
from itertools import product
from typing import Dict
import numpy as np
from line_profiler import profile

class EwaldPotential(nn.Module):
    def __init__(self,
                 dl=4.0,  # grid resolution
                 sigma=5.0,  # width of the Gaussian on each atom
                 remove_self_interaction=False,
                 aggregation_mode: str = "sum"):
        
        super().__init__()
        self.dl = dl
        self.sigma = sigma
        self.sigma_sq_half = sigma ** 2 / 2.0
        self.twopi = 2.0 * torch.pi
        self.twopi_sq = self.twopi ** 2
        self.remove_self_interaction = remove_self_interaction
        # self.feature_key = feature_key
        # self.output_key = output_key
        self.aggregation_mode = aggregation_mode
        # self.model_outputs = [output_key]
        self.norm_factor = 1.0
        self.k_sq_max = (self.twopi / self.dl) ** 2

    def forward(self, q_vector, k_vector, v_vector, data: Dict[str, torch.Tensor], **kwargs):
        print("self.dl k-grid:", self.dl)
        print("q_vector from forward:", q_vector.shape)
        print("torch.isfinite(q_vector).all():", torch.isfinite(q_vector).all(), "torch.isnan(q_vector).any():", torch.isnan(q_vector).all())
        print("torch.isfinite(k_vector).all():", torch.isfinite(k_vector).all(), "torch.isnan(k_vector).any():", torch.isnan(k_vector).all())
        print("torch.isfinite(v_vector).all():", torch.isfinite(v_vector).all(), "torch.isnan(v_vector).any():", torch.isnan(v_vector).all())

        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch_now = data["batch"]
        
        box = data['cell'].view(-1, 3, 3).diagonal(dim1=-2, dim2=-1)
        r = data['positions']

        n, d = r.shape
        assert d == 3, 'r dimension error'
        assert n == q_vector.size(0), 'q dimension error'
        assert n == k_vector.size(0), 'k dimension error'
        assert n == v_vector.size(0), 'v dimension error'

        unique_batches = torch.unique(batch_now)  # Get unique batch indices

        results = []
        for i in unique_batches:
            mask = batch_now == i  # Create a mask for the i-th configuration
            # Calculate the potential energy for the i-th configuration
            k_pot = self.compute_potential_optimized(r[mask], k_vector[mask], box[i], 'k')
            v_pot = self.compute_potential_optimized(r[mask], v_vector[mask], box[i], 'v')
            q_pot = self.compute_potential_optimized(r[mask], q_vector[mask], box[i], 'q')
            print("q_pot:", q_pot.shape, k_pot.shape, v_pot.shape)

            attention_weights = torch.einsum('ijk,jk->ij', torch.transpose(torch.abs(q_pot), 1, 2), torch.abs(k_pot))
            # print("i:", i, "q_pot dtype:", q_pot.dtype, "torch.isfinite(q_pot).all():", torch.isfinite(q_pot).all(), "torch.isnan(q_pot).any():", torch.isnan(q_pot).all(), "min. {:.5f}, max. {:.5f}".format(q_pot.real.min(), q_pot.real.max()))
            # print("i:", i, "k_pot dtype:", k_pot.dtype, "torch.isfinite(k_pot).all():", torch.isfinite(k_pot).all(), "torch.isnan(k_pot).any():", torch.isnan(k_pot).all(), "min. {:.5f}, max. {:.5f}".format(k_pot.real.min(), k_pot.real.max()))
            # print("i:", i, "v_pot dtype:", v_pot.dtype, "torch.isfinite(v_pot).all():", torch.isfinite(v_pot).all(), "torch.isnan(v_pot).any():", torch.isnan(v_pot).all(), "min. {:.5f}, max. {:.5f}".format(v_pot.real.min(), v_pot.real.max()))

            real_attention_weights = torch.real(attention_weights)
            # attention_weights = attention_weights / torch.sqrt(torch.tensor(k_pot.shape[1], dtype=torch.float))
            print("i:", i, "attention_weights dtype:", attention_weights.dtype, "torch.isfinite(attention_weights).all():", torch.isfinite(attention_weights).all(), "torch.isnan(attention_weights).any():", torch.isnan(attention_weights).all(), "min. {:.5f}, max. {:.5f}".format(attention_weights.real.min(), attention_weights.real.max()))

            max_values, _ = torch.max(real_attention_weights, dim=-1, keepdim=True)
            shifted_attention_weights = real_attention_weights - max_values

            log_attention_weights = torch.log_softmax(shifted_attention_weights, dim=-1)
            softmax_attention_weights = torch.exp(log_attention_weights)

            weighted_values = softmax_attention_weights[:, :, None] * v_pot[None, :, :]
            real_space_weighted_values = self.compute_inverse_transform_optimized(r[mask], weighted_values, box[i])

            print("node_feats_list:", weighted_values.shape, real_space_weighted_values.shape)
            results.append(torch.real(real_space_weighted_values))

        results = torch.cat(results, dim=0) 
        print("results:", results.shape)       
        return results

    def compute_potential_optimized(self, r_raw, q, box, vector_type):
        dtype = torch.complex64 if r_raw.dtype == torch.float32 else torch.complex128
        device = r_raw.device

        r = r_raw / box

        nk = (box / self.dl).int().tolist()
        nk = [max(1, k) for k in nk]

        n = r.shape[0]
        eikx = torch.ones((n, nk[0] + 1), dtype=dtype, device=device)
        eiky = torch.ones((n, 2 * nk[1] + 1), dtype=dtype, device=device)
        eikz = torch.ones((n, 2 * nk[2] + 1), dtype=dtype, device=device)

        eikx[:, 1] = torch.exp(1j * self.twopi * r[:, 0])
        eiky[:, nk[1] + 1] = torch.exp(1j * self.twopi * r[:, 1])
        eikz[:, nk[2] + 1] = torch.exp(1j * self.twopi * r[:, 2])

        for k in range(2, nk[0] + 1):
            eikx[:, k] = eikx[:, k - 1].clone() * eikx[:, 1].clone()
        for k in range(2, nk[1] + 1):
            eiky[:, nk[1] + k] = eiky[:, nk[1] + k - 1].clone() * eiky[:, nk[1] + 1].clone()
        for k in range(2, nk[2] + 1):
            eikz[:, nk[2] + k] = eikz[:, nk[2] + k - 1].clone() * eikz[:, nk[2] + 1].clone()

        for k in range(nk[1]):
            eiky[:, k] = torch.conj(eiky[:, 2 * nk[1] - k])
        for k in range(nk[2]):
            eikz[:, k] = torch.conj(eikz[:, 2 * nk[2] - k])

        kx = torch.arange(nk[0] + 1, device=device)
        ky = torch.arange(-nk[1], nk[1] + 1, device=device)
        kz = torch.arange(-nk[2], nk[2] + 1, device=device)

        kx_term = (kx / box[0]) ** 2
        ky_term = (ky / box[1]) ** 2
        kz_term = (kz / box[2]) ** 2

        kx_sq = kx_term.view(-1, 1, 1)
        ky_sq = ky_term.view(1, -1, 1)
        kz_sq = kz_term.view(1, 1, -1)

        k_sq = self.twopi_sq * (kx_sq + ky_sq + kz_sq)
        mask = (k_sq <= self.k_sq_max) & (k_sq > 0)

        kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
        kfac[~mask] = 0

        eikx_expanded = eikx.unsqueeze(2).unsqueeze(3)
        eiky_expanded = eiky.unsqueeze(1).unsqueeze(3)
        eikz_expanded = eikz.unsqueeze(1).unsqueeze(2)

        factor = torch.ones_like(kx, dtype=dtype, device=device)
        factor[1:] = 2.0

        if q.dim() == 1:
            q_expanded = q.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        elif q.dim() == 2:
            q_expanded = q.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        else:
            raise ValueError("q must be 1D or 2D tensor")
        
        k_vectors = mask.nonzero()
        print("box size:", box)
        print("k_vectors shape and dtype:", k_vectors.shape, k_vectors.dtype)

        value = q_expanded * (eikx_expanded * eiky_expanded * eikz_expanded).unsqueeze(1)
        print("value from optimized ewald sum:", value.shape)
        k_vectors_adjusted = k_vectors - 1
        
        # term = torch.stack([value[:, :, kx-1, ky-1, kz-1].sum(dim=0) for kx, ky, kz in k_vectors])
        # term = term.transpose(0, 1)

        term_before_sum = value[:, :, k_vectors_adjusted[:, 0], k_vectors_adjusted[:, 1], k_vectors_adjusted[:, 2]]
        # term_before_sum = value[:, :, k_vectors[:, 0], k_vectors[:, 1], k_vectors[:, 2]]
        
        term = term_before_sum.sum(dim=0)
        term = term.transpose(0, 1)

        if vector_type == 'v' or vector_type == 'k':
            print("term from value or key:", term_before_sum.shape, value.shape, term.shape)
            return_val = term
        else:
            print("term from query:", term_before_sum.shape, value.shape, term.shape)
            return_val = term_before_sum

        # pot = (kfac.unsqueeze(0) * factor.view(1, -1, 1, 1) * torch.real(torch.conj(term) * term)).sum(dim=[1, 2, 3])

        # pot /= (box[0] * box[1] * box[2])

        # if self.remove_self_interaction:
        #     pot -= torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.))

        return return_val #pot.real * self.norm_factor

    @profile
    def compute_inverse_transform_optimized(self, r_raw, q, box):
        dtype = torch.complex64 if r_raw.dtype == torch.float32 else torch.complex128
        device = r_raw.device

        r = r_raw / box
        print("optimized_inverse:", r_raw.shape, r.shape)

        nk = (box / self.dl).int().tolist()
        nk = [max(1, k) for k in nk]

        n = r.shape[0]
        eikx = torch.ones((n, nk[0] + 1), dtype=dtype, device=device)
        eiky = torch.ones((n, 2 * nk[1] + 1), dtype=dtype, device=device)
        eikz = torch.ones((n, 2 * nk[2] + 1), dtype=dtype, device=device)

        eikx[:, 1] = torch.exp(-1j * self.twopi * r[:, 0])
        eiky[:, nk[1] + 1] = torch.exp(-1j * self.twopi * r[:, 1])
        eikz[:, nk[2] + 1] = torch.exp(-1j * self.twopi * r[:, 2])

        for k in range(2, nk[0] + 1):
            eikx[:, k] = eikx[:, k - 1].clone() * eikx[:, 1].clone()
        for k in range(2, nk[1] + 1):
            eiky[:, nk[1] + k] = eiky[:, nk[1] + k - 1].clone() * eiky[:, nk[1] + 1].clone()
        for k in range(2, nk[2] + 1):
            eikz[:, nk[2] + k] = eikz[:, nk[2] + k - 1].clone() * eikz[:, nk[2] + 1].clone()

        for k in range(nk[1]):
            eiky[:, k] = torch.conj(eiky[:, 2 * nk[1] - k])
        for k in range(nk[2]):
            eikz[:, k] = torch.conj(eikz[:, 2 * nk[2] - k])

        kx_vals = torch.arange(0, nk[0] + 1, device=device)
        ky_vals = torch.arange(-nk[1], nk[1] + 1, device=device)
        kz_vals = torch.arange(-nk[2], nk[2] + 1, device=device)

        kx_grid, ky_grid, kz_grid = torch.meshgrid(kx_vals, ky_vals, kz_vals, indexing='ij')
        print("mesh grid shape:", kx_grid.shape, ky_grid.shape, kz_grid.shape)

        k_sq_grid = self.twopi_sq * ((kx_grid / box[0])**2 + (ky_grid / box[1])**2 + (kz_grid / box[2])**2)
        valid_k = (k_sq_grid <= self.k_sq_max) & (k_sq_grid > 0)  # Apply the condition for k_sq_max and remove k=0 term
        print("valid_k:", valid_k.shape)

        kx_valid = kx_grid[valid_k]
        ky_valid = ky_grid[valid_k]
        kz_valid = kz_grid[valid_k]
        k_sq_valid = k_sq_grid[valid_k]

        eikx_valid = eikx[:, kx_valid]
        eiky_valid = eiky[:, nk[1] + ky_valid]
        eikz_valid = eikz[:, nk[2] + kz_valid]
        
        q_valid = q[:, :len(kx_valid), :]
        print("q, torch.isfinite(q).all():", torch.isfinite(q).all(), "torch.isnan(q).any():", torch.isnan(q).all(), "min. {:.5f}, max. {:.5f}".format(q.real.min(), q.real.max()))
        print("eikx_valid, torch.isfinite(eikx_valid).all():", torch.isfinite(eikx_valid).all(), "torch.isnan(eikx_valid).any():", torch.isnan(eikx_valid).all(), "min. {:.5f}, max. {:.5f}".format(eikx_valid.real.min(), eikx_valid.real.max()))
        print("eiky_valid, torch.isfinite(eiky_valid).all():", torch.isfinite(eiky_valid).all(), "torch.isnan(eiky_valid).any():", torch.isnan(eiky_valid).all(), "min. {:.5f}, max. {:.5f}".format(eiky_valid.real.min(), eiky_valid.real.max()))
        print("eikz_valid, torch.isfinite(eikz_valid).all():", torch.isfinite(eikz_valid).all(), "torch.isnan(eikz_valid).any():", torch.isnan(eikz_valid).all(), "min. {:.5f}, max. {:.5f}".format(eikz_valid.real.min(), eikz_valid.real.max()))

        term_before_sum = q.unsqueeze(2) * eikx_valid.unsqueeze(2).unsqueeze(3) * eiky_valid.unsqueeze(2).unsqueeze(3) * eikz_valid.unsqueeze(2).unsqueeze(3)
        new_sum_term = torch.sum(term_before_sum, dim=1).reshape(n, -1)

        print("new_sum_term:", new_sum_term.shape, n) #[atoms, features]

        # kfac = torch.exp(-sigma_sq_half * k_sq) / k_sq
        # kfac[~mask] = 0

        # eikx_expanded = eikx.unsqueeze(2).unsqueeze(3)
        # eiky_expanded = eiky.unsqueeze(1).unsqueeze(3)
        # eikz_expanded = eikz.unsqueeze(1).unsqueeze(2)

        # factor = torch.ones_like(kx, dtype=dtype, device=device)
        # factor[1:] = 2.0

        # if q.dim() == 1:
        #     q_expanded = q.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        # elif q.dim() == 2:
        #     q_expanded = q.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        # else:
        #     raise ValueError("q must be 1D or 2D tensor")

        # k_vectors = mask.nonzero()
        # print("eikx, eiky, eikz shape:", eikx.shape, eiky.shape, eikz.shape, k_sq.shape,  mask.nonzero().shape)
        # print("k_vectors shape:", k_vectors.shape, torch.max(k_vectors, dim=0), torch.min(k_vectors, dim=0))

        # value = q_expanded * (eikx_expanded * eiky_expanded * eikz_expanded).unsqueeze(1)
        # term_before_sum = q_valid.unsqueeze(2) * eikx_valid.unsqueeze(2).unsqueeze(3) * eiky_valid.unsqueeze(2).unsqueeze(3) * eikz_valid.unsqueeze(2).unsqueeze(3)
        # new_sum_term = torch.sum(term_before_sum, dim=1)

        # print("value from optimized ewald sum:", term_before_sum.shape, new_sum_term.shape)

        # k_vectors_adjusted = k_vectors - 1
        # term = torch.stack([value[:, :, kx-1, ky-1, kz-1].sum(dim=0) for kx, ky, kz in k_vectors])
        # term = term.transpose(0, 1)
        # term_before_sum = value[:, :, k_vectors_adjusted[:, 0], k_vectors_adjusted[:, 1], k_vectors_adjusted[:, 2]]
        # term = value.sum(dim=[2, 3, 4])
        # term = term.transpose(0, 1)

        return_val = new_sum_term

        return return_val