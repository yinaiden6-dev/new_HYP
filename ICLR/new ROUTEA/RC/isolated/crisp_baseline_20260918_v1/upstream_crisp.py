"""CRISP scoring: Cross-directional Refined Informative Scoring of Patches."""

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def patch_weights(img_embs: list[torch.Tensor]) -> list[torch.Tensor]:
    """w_s = d_s / sum(d) * |A|, d_s = || p_s - mean_active ||_2 ; 0 on padding."""
    out = []
    for t in img_embs:
        norms = t.norm(dim=1)
        mask = norms > 1e-6
        if mask.sum() <= 1:
            out.append(torch.ones(t.size(0)))
            continue
        active = t[mask]
        mean = active.mean(dim=0, keepdim=True)
        dist = torch.zeros(t.size(0))
        dist[mask] = (active - mean).norm(dim=1)
        total = dist.sum().clamp(min=1e-8)
        w = dist / total * mask.sum().float()
        w[~mask] = 0.0
        out.append(w)
    return out


def crisp_score(txt_embs: list[torch.Tensor],
                img_embs: list[torch.Tensor],
                device: torch.device,
                batch_size: int) -> torch.Tensor:
    """Compute the (n_query, n_image) CRISP score matrix on `device`."""
    pw_list = patch_weights(img_embs)
    n_q, n_img = len(txt_embs), len(img_embs)
    scores = torch.zeros(n_q, n_img)

    for qi in tqdm(range(0, n_q, batch_size), desc="  CRISP scoring"):
        qe = min(qi + batch_size, n_q)
        qs = pad_sequence(txt_embs[qi:qe], batch_first=True,
                          padding_value=0).to(device)
        qs_mask = qs.abs().sum(-1) > 1e-6

        parts = []
        for ii in range(0, n_img, batch_size):
            ie = min(ii + batch_size, n_img)
            ps = pad_sequence(img_embs[ii:ie], batch_first=True,
                              padding_value=0).to(device)
            ps_mask = ps.abs().sum(-1) > 1e-6
            pw = pad_sequence(pw_list[ii:ie], batch_first=True,
                              padding_value=0).to(device)

            sim = torch.einsum("bnd,csd->bcns", qs, ps)

            # Reverse pass first (sim is modified in-place afterwards).
            inv_q = (~qs_mask).unsqueeze(1).unsqueeze(3)
            sim.masked_fill_(inv_q, float("-inf"))
            rev = sim.max(dim=2)[0]
            sim.masked_fill_(inv_q, 0)
            rev.masked_fill_(~ps_mask.unsqueeze(0), 0)
            rev_sum = rev.sum(dim=2)
            del rev, inv_q

            # Forward adjusted MaxSim with patch weights (in-place).
            pw_4d = pw.unsqueeze(0).unsqueeze(2)
            sim.mul_(pw_4d)
            fwd_sum_raw = sim.sum(dim=3)
            valid = (pw_4d > 0).float().sum(dim=3)
            sim.masked_fill_(pw_4d == 0, float("-inf"))
            fwd_max = sim.max(dim=3)[0]
            fwd_mean = fwd_sum_raw / valid.clamp(min=1)
            fwd = fwd_max - fwd_mean
            fwd.masked_fill_(~qs_mask.unsqueeze(1), 0)
            fwd_sum = fwd.sum(dim=2)
            del sim, pw_4d, fwd_sum_raw, valid, fwd_max, fwd_mean, fwd

            n_qt = qs_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            n_pt = ps_mask.sum(dim=1, keepdim=True).float().clamp(min=1).T
            parts.append((fwd_sum / n_qt + rev_sum / n_pt).cpu())
            del ps, pw, fwd_sum, rev_sum

        scores[qi:qe] = torch.cat(parts, dim=1)
    return scores
