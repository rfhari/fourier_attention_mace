def compute_inverse_transform_optimized_v1(self, r_raw, q, box):
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
    print("eikx, eiky, eikz shape:", eikx.shape, eiky.shape, eikz.shape, k_sq.shape,  mask.nonzero().shape)
    print("k_vectors shape:", k_vectors.shape, torch.max(k_vectors, dim=0), torch.min(k_vectors, dim=0))

    value = q_expanded * (eikx_expanded * eiky_expanded * eikz_expanded).unsqueeze(1)
    print("value from optimized ewald sum:", value.shape)
    k_vectors_adjusted = k_vectors - 1
    # term = torch.stack([value[:, :, kx-1, ky-1, kz-1].sum(dim=0) for kx, ky, kz in k_vectors])
    # term = term.transpose(0, 1)
    term_before_sum = value[:, :, k_vectors_adjusted[:, 0], k_vectors_adjusted[:, 1], k_vectors_adjusted[:, 2]]
    term = value.sum(dim=[2, 3, 4])
    # term = term.transpose(0, 1)

    print("term from value or key:", value.shape, term.shape)
    return_val = term

    # pot = (kfac.unsqueeze(0) * factor.view(1, -1, 1, 1) * torch.real(torch.conj(term) * term)).sum(dim=[1, 2, 3])

    # pot /= (box[0] * box[1] * box[2])

    # if self.remove_self_interaction:
    #     pot -= torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.))

    return return_val #pot.real * self.norm_factor
