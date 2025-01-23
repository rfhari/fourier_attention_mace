import torch
import torch.nn as nn
from itertools import product
from typing import Dict
import numpy as np
from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from e3nn import o3

class EwaldPotential(nn.Module):
    def __init__(self,
                 dl=4.0,  # grid resolution
                 sigma=5.0,  # width of the Gaussian on each atom
                 remove_self_interaction=False,
                #  feature_key: str = 'q',
                #  output_key: str = 'ewald_potential',
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
        self.num_heads = 4
        self.embed_dim = 256
        assert self.embed_dim % self.num_heads == 0, "wrong embed dim and num_heads"
        self.head_dim = self.embed_dim//self.num_heads
        self.output_dim = str(3*self.embed_dim)
        self.ewald_readout = LinearReadoutBlock(o3.Irreps(str(self.embed_dim) + "x0e"), o3.Irreps(self.output_dim + "x0e"))    
        # self.q_readout = LinearReadoutBlock(o3.Irreps("256x0e"), o3.Irreps("256x0e"))     
        # self.v_readout = LinearReadoutBlock(o3.Irreps("256x0e"), o3.Irreps("256x0e"))     
        # self.k_readout = LinearReadoutBlock(o3.Irreps("256x0e"), o3.Irreps("256x0e"))

    def forward(self, x_vector, data: Dict[str, torch.Tensor], **kwargs):
        # print("q_vector from forward:", q_vector.shape)
        # print("torch.isfinite(q_vector).all():", torch.isfinite(q_vector).all(), "torch.isnan(q_vector).any():", torch.isnan(q_vector).all())
        # print("torch.isfinite(k_vector).all():", torch.isfinite(k_vector).all(), "torch.isnan(k_vector).any():", torch.isnan(k_vector).all())
        # print("torch.isfinite(v_vector).all():", torch.isfinite(v_vector).all(), "torch.isnan(v_vector).any():", torch.isnan(v_vector).all())
        qkv = self.ewald_readout(x_vector)
        
        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch_now = data["batch"]
        
        box = data['cell'].view(-1, 3, 3).diagonal(dim1=-2, dim2=-1)
        r = data['positions']
        qkv = qkv.reshape(len(batch_now), self.num_heads, 3*self.head_dim)
        print("qkv:", qkv.shape, r.shape)

        # q = data[self.feature_key]
        # if q.dim() == 1:
        #     q = q.unsqueeze(1)

        n, d = r.shape
        assert d == 3, 'r dimension error'

        unique_batches = torch.unique(batch_now)  

        results = []
        for i in unique_batches:
            mask = batch_now == i  # Create a mask for the i-th configuration
            qkv_masked = qkv[mask].permute(1, 0, 2)   
            q_vector_mha, k_vector_mha, v_vector_mha = qkv_masked.chunk(3, dim=-1)     
            print("mha vectors after split:", q_vector_mha.shape, k_vector_mha.shape, v_vector_mha.shape)
            # all_head_weighted_values = []

            # for head_ind in range(q_vector_mha.shape[0]):
            #     q_vector, k_vector, v_vector = q_vector_mha[head_ind, :, :], k_vector_mha[head_ind, :, :], v_vector_mha[head_ind, :, :]
            #     print("single head vectors after split:", q_vector.shape, k_vector.shape, v_vector.shape)
            #     print("qkv after permute:", qkv.shape)    

            #     k_pot = self.compute_potential_optimized(r[mask], k_vector, box[i], 'k')
            #     v_pot = self.compute_potential_optimized(r[mask], v_vector, box[i], 'v')
            #     q_pot = self.compute_potential_optimized(r[mask], q_vector, box[i], 'q')
            #     print("q_pot in each head:", q_pot.shape, k_pot.shape, v_pot.shape)

            #     attention_weights = torch.einsum('ijk,jk->ij', torch.transpose(q_pot, 1, 2), k_pot)

            #     real_attention_weights = torch.real(attention_weights)

            #     max_values, _ = torch.max(real_attention_weights, dim=-1, keepdim=True)
            #     shifted_attention_weights = real_attention_weights - max_values

            #     softmax_attention_weights = torch.softmax(shifted_attention_weights, dim=-1)

            #     weighted_values = softmax_attention_weights[:, :, None] * v_pot[None, :, :]
            #     all_head_weighted_values.append(weighted_values)

            # all_head_weighted_values = torch.cat(all_head_weighted_values, dim=-1)
            # print("weighted_values before ift:", all_head_weighted_values.shape, softmax_attention_weights.shape, weighted_values.shape)

# ---------------------- efficient implementation -------------------------------
            num_heads, n_atoms, head_dim = q_vector_mha.shape
            
            q_vector_flat = q_vector_mha.reshape(-1, head_dim)
            k_vector_flat = k_vector_mha.reshape(-1, head_dim)
            v_vector_flat = v_vector_mha.reshape(-1, head_dim)
            print("flattened vector:", q_vector_flat.shape, k_vector_flat.shape, v_vector_flat.shape)

            k_pot = self.compute_potential_optimized(r[mask], k_vector_flat, box[i], 'k', num_heads)
            v_pot = self.compute_potential_optimized(r[mask], v_vector_flat, box[i], 'v', num_heads)
            q_pot = self.compute_potential_optimized(r[mask], q_vector_flat, box[i], 'q', num_heads)
            print("efficient implementation q_pot:", q_pot.shape, k_pot.shape, v_pot.shape)

            efficient_attention_weights = torch.einsum('hijk,hjk->hij', torch.abs(q_pot).transpose(2, 3), torch.abs(k_pot))
            efficient_real_attention_weights = torch.real(efficient_attention_weights)

            efficient_max_values, _ = torch.max(efficient_real_attention_weights, dim=-1, keepdim=True)
            efficient_shifted_attention_weights = efficient_real_attention_weights - efficient_max_values
            efficient_softmax_attention_weights = torch.softmax(efficient_shifted_attention_weights, dim=-1)
            efficient_weighted_values = efficient_softmax_attention_weights.unsqueeze(-1) * v_pot.unsqueeze(1)
            efficient_weighted_values = efficient_weighted_values.permute(1, 2, 0, 3).reshape(n_atoms, k_pot.shape[1], -1) 

            print("efficient_weighted_values:", efficient_weighted_values.shape)
# ---------------------- efficient implementation -------------------------------

            # print("weighted_values before ift:", all_head_weighted_values.shape, efficient_weighted_values.shape)
            # print("weighted_values comparison before ift:", torch.allclose(all_head_weighted_values, efficient_weighted_values))
            real_space_weighted_values = self.compute_inverse_transform_optimized(r[mask], efficient_weighted_values, box[i])
            print("weighted_values after ift:", real_space_weighted_values.shape)

            results.append(torch.real(real_space_weighted_values))

        results = torch.cat(results, dim=0) 
        print("results:", results.shape)       
        return results

    def compute_potential(self, r_raw, q, box, value):
        dtype = torch.complex64 if r_raw.dtype == torch.float32 else torch.complex128
        device = r_raw.device

        r = r_raw / box  # Work with scaled positions
        # r =  r - torch.round(r) # periodic boundary condition

        # Calculate nk based on the provided box dimensions and resolution
        nk = (box / self.dl).int().tolist()
        print("nk:", nk)
        
        for i in range(3):
            if nk[i] < 1: nk[i] = 1
        
        n = r.shape[0]
        eikx = torch.zeros((n, nk[0] + 1), dtype=dtype, device=device)
        eiky = torch.zeros((n, 2 * nk[1] + 1), dtype=dtype, device=device)
        eikz = torch.zeros((n, 2 * nk[2] + 1), dtype=dtype, device=device)

        eikx[:, 0] = torch.ones(n, dtype=dtype, device=device)
        eiky[:, nk[1]] = torch.ones(n, dtype=dtype, device=device)
        eikz[:, nk[2]] = torch.ones(n, dtype=dtype, device=device)

        # Calculate remaining positive kx, ky, and kz terms by recursion
        for k in range(1, nk[0] + 1):
            eikx[:, k] = torch.exp(-1j * self.twopi * k * r[:, 0]) 
        for k in range(1, nk[1] + 1):
            eiky[:, nk[1] + k] = torch.exp(-1j * self.twopi * k * r[:, 1])
        for k in range(1, nk[2] + 1):
            eikz[:, nk[2] + k] = torch.exp(-1j * self.twopi * k * r[:, 2])

        # Negative k values are complex conjugates of positive ones
        for k in range(nk[1]):
            eiky[:, k] = torch.conj(eiky[:, 2 * nk[1] - k])
        for k in range(nk[2]):
            eikz[:, k] = torch.conj(eikz[:, 2 * nk[2] - k])

        pot_list, term_list, counter = [], [], 0
        print("kx, ky, kz", eikx.shape, eiky.shape, eikz.shape)
        for kx in range(nk[0] + 1):
            factor = 1.0 if kx == 0 else 2.0
            for ky, kz in product(range(-nk[1], nk[1] + 1), range(-nk[2], nk[2] + 1)):
                k_sq = self.twopi_sq * ((kx / box[0]) ** 2 + (ky / box[1]) ** 2 + (kz / box[2]) ** 2)
                if k_sq <= self.k_sq_max and k_sq > 0:  # remove the k=0 term
                    kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
                    term = torch.sum(q * (eikx[:, kx].unsqueeze(1) * eiky[:, nk[1] + ky].unsqueeze(1) * eikz[:, nk[2] + kz].unsqueeze(1)), dim=0)
                    term_before_sum = q * (eikx[:, kx].unsqueeze(1) * eiky[:, nk[1] + ky].unsqueeze(1) * eikz[:, nk[2] + kz].unsqueeze(1))
                    term_list.append(term_before_sum)
                    counter+=1
                    # pot_list.append(factor * kfac * torch.real(torch.conj(term) * term))
        term_list = torch.stack(term_list, dim=0) #shape of [k-vectors, atoms, features]
        
        if value=='v' or value == 'k':
            sum_term = torch.sum(term_list, dim=1)
        else:
            sum_term = term_list
        
        print("values in FT block:", counter, term.shape, term_before_sum.shape, term_list.shape, sum_term.shape)   
        # print(torch.real(term))
        # pot = torch.stack(pot_list).sum(axis=0) / (box[0] * box[1] * box[2])
        

        # if self.remove_self_interaction:
        #     pot -= torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.))

        return sum_term #torch.real(torch.conj(term) * term)
    
    def compute_inverse_transform(self, r_raw, q, box):
        # r_raw = r_raw.reshape(1, -1)
        dtype = torch.complex64 if r_raw.dtype == torch.float32 else torch.complex128
        device = r_raw.device

        r = r_raw / box  # Work with scaled positions
        # r =  r - torch.round(r) # periodic boundary condition

        # Calculate nk based on the provided box dimensions and resolution
        nk = (box / self.dl).int().tolist()
        print("IFT nk:", nk, r.shape, q.shape, box, self.dl, self.k_sq_max)
        
        for i in range(3):
            if nk[i] < 1: nk[i] = 1
        
        n = r.shape[0]
        eikx = torch.zeros((n, nk[0] + 1), dtype=dtype, device=device)
        eiky = torch.zeros((n, 2 * nk[1] + 1), dtype=dtype, device=device)
        eikz = torch.zeros((n, 2 * nk[2] + 1), dtype=dtype, device=device)

        eikx[:, 0] = torch.ones(n, dtype=dtype, device=device)
        eiky[:, nk[1]] = torch.ones(n, dtype=dtype, device=device)
        eikz[:, nk[2]] = torch.ones(n, dtype=dtype, device=device)

        for k in range(1, nk[0] + 1):
            eikx[:, k] = torch.exp(1j * self.twopi * k * r[:, 0]) 
        for k in range(1, nk[1] + 1):
            eiky[:, nk[1] + k] = torch.exp(1j * self.twopi * k * r[:, 1])
        for k in range(1, nk[2] + 1):
            eikz[:, nk[2] + k] = torch.exp(1j * self.twopi * k * r[:, 2])

        for k in range(nk[1]):
            eiky[:, k] = torch.conj(eiky[:, 2 * nk[1] - k])
        for k in range(nk[2]):
            eikz[:, k] = torch.conj(eikz[:, 2 * nk[2] - k])

        pot_list, term_list, counter = [], [], 0
        print("IFT kx, ky, kz", eikx.shape, q[counter, :].shape, eiky.shape, eikz.shape)
        for kx in range(nk[0] + 1):
            factor = 1.0 if kx == 0 else 2.0
            # print("IFT kx:", eikx[:, kx].unsqueeze(1).shape)
            for ky, kz in product(range(-nk[1], nk[1] + 1), range(-nk[2], nk[2] + 1)):
                k_sq = self.twopi_sq * ((kx / box[0]) ** 2 + (ky / box[1]) ** 2 + (kz / box[2]) ** 2)
                if k_sq <= self.k_sq_max and k_sq > 0:  # remove the k=0 term
                    # kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
                    term = torch.sum(q[counter, :].reshape(-1) * (eikx[:, kx].unsqueeze(1) * eiky[:, nk[1] + ky].unsqueeze(1) * eikz[:, nk[2] + kz].unsqueeze(1)), dim=0)
                    term_before_sum = q[counter, :].reshape(-1) * (eikx[:, kx].unsqueeze(1) * eiky[:, nk[1] + ky].unsqueeze(1) * eikz[:, nk[2] + kz].unsqueeze(1))
                    term_list.append(term_before_sum)
                    counter+=1
                    # pot_list.append(factor * kfac * torch.real(torch.conj(term) * term))
        term_list = torch.stack(term_list, dim=0) # shape of [k-vectors, atoms, features]
        sum_term = torch.sum(term_list, dim=0)
        print("values in IFT block:", counter, term.shape, term_before_sum.shape, term_list.shape, sum_term.shape)

        # print("IFT sum term:", sum_term)   
        # print(torch.real(term))
        # pot = torch.stack(pot_list).sum(axis=0) / (box[0] * box[1] * box[2])
        

        # if self.remove_self_interaction:
        #     pot -= torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.))

        return torch.real(sum_term2) #torch.real(torch.conj(term) * term)

    # Optimized function
    def compute_potential_optimized(self, r_raw, q, box, vector_type, num_heads=1):
        dtype = torch.complex64 if r_raw.dtype == torch.float32 else torch.complex128
        device = r_raw.device

        r = r_raw / box
        r = r.repeat(num_heads, 1)

        nk = (box / self.dl).int().tolist()
        nk = [max(1, k) for k in nk]

        n = r.shape[0]
        print("num_heads and n:", num_heads, n)

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

        # kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
        # kfac[~mask] = 0

        eikx_expanded = eikx.unsqueeze(2).unsqueeze(3)
        eiky_expanded = eiky.unsqueeze(1).unsqueeze(3)
        eikz_expanded = eikz.unsqueeze(1).unsqueeze(2)

        # factor = torch.ones_like(kx, dtype=dtype, device=device)
        # factor[1:] = 2.0

        if q.dim() == 1:
            q_expanded = q.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        elif q.dim() == 2:
            q_expanded = q.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        else:
            raise ValueError("q must be 1D or 2D tensor")
        
        k_vectors = mask.nonzero()
        print("k_vectors, q_expanded and eikx_expanded shape:", k_vectors.shape, q_expanded.shape, eikx_expanded.shape, eiky_expanded.shape, eikz_expanded.shape)

        value = q_expanded * (eikx_expanded * eiky_expanded * eikz_expanded).unsqueeze(1)
        print("value from optimized ewald sum:", value.shape)
        k_vectors_adjusted = k_vectors - 1
        
        # term = torch.stack([value[:, :, kx-1, ky-1, kz-1].sum(dim=0) for kx, ky, kz in k_vectors])
        # term = term.transpose(0, 1)

        term_before_sum = value[:, :, k_vectors_adjusted[:, 0], k_vectors_adjusted[:, 1], k_vectors_adjusted[:, 2]]
        # term_before_sum = value[:, :, k_vectors[:, 0], k_vectors[:, 1], k_vectors[:, 2]]
        
        if num_heads>1:
            if vector_type == 'v' or vector_type == 'k':
                term_before_sum = term_before_sum.view(num_heads, r_raw.shape[0], *term_before_sum.shape[1:])
                term = term_before_sum.sum(dim=1)
                term = term.transpose(1, 2)
                print("term from value or key:", term_before_sum.shape, value.shape, term.shape)
                return_val = term
            else:
                print("term from query:", term_before_sum.shape, value.shape)
                return_val = term_before_sum.view(num_heads, r_raw.shape[0], *term_before_sum.shape[1:])            
        else:
            if vector_type == 'v' or vector_type == 'k':
                term = term_before_sum.sum(dim=0)
                term = term.transpose(0, 1)
                print("term from value or key:", term_before_sum.shape, value.shape, term.shape)
                return_val = term
            else:
                print("term from query:", term_before_sum.shape, value.shape)
                return_val = term_before_sum

        return return_val 

    def compute_inverse_transform_optimized(self, r_raw, q, box):
        # r_raw = r_raw.reshape(1,-1)
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

        # Calculate remaining positive kx, ky, and kz terms by recursion
        for k in range(2, nk[0] + 1):
            eikx[:, k] = eikx[:, k - 1].clone() * eikx[:, 1].clone()
        for k in range(2, nk[1] + 1):
            eiky[:, nk[1] + k] = eiky[:, nk[1] + k - 1].clone() * eiky[:, nk[1] + 1].clone()
        for k in range(2, nk[2] + 1):
            eikz[:, nk[2] + k] = eikz[:, nk[2] + k - 1].clone() * eikz[:, nk[2] + 1].clone()

        # Negative k values are complex conjugates of positive ones
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

        print("shapes of q:", q.shape, torch.allclose(q_valid, q))

        term_before_sum = q.unsqueeze(2) * eikx_valid.unsqueeze(2).unsqueeze(3) * eiky_valid.unsqueeze(2).unsqueeze(3) * eikz_valid.unsqueeze(2).unsqueeze(3)

        new_sum_term = torch.sum(term_before_sum, dim=1).reshape(n, -1)

        print("new_sum_term:", new_sum_term.shape, n) #[atoms, features]

        return_val = new_sum_term

        return return_val