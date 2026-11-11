#!/usr/bin/env python3
"""TrainRecords: the 112-byte records of the engine's self-play (src/selfplay.cpp), as the trainers read them.

  REC            numpy dtype of one record (board, side to move, urns, score, result, flags, ply, move, no-capture count)
  keep_mask      the training filter (no position in check, no capture unless asked, no mate score, no damaged record)
  load_records   filtered records of files, in one array
  fields         a batch of raw records (torch tensor) -> board, stm, urn, score, result, flags
  fit_k          the score scale K of P(win) ~ sigmoid(score / K)
"""
import numpy as np
import torch

REC = np.dtype([("board", "u1", (90,)), ("stm", "u1"), ("urn", "i1", (2, 6)), ("score", "<i2"), ("result", "i1"),
                ("flags", "u1"), ("ply", "<u2"), ("move", "<u2"), ("nocap", "u1")])
assert REC.itemsize == 112
URN_INIT = [2, 2, 2, 2, 2, 5]


def keep_mask(a, keep_captures=False, min_ply=0):
    """Training filter on (N,112) records: no in-check (flag 1), no captures (flag 2) unless asked,
    ply >= min_ply, no mate / known-win scores."""
    flags = a[:, 106]
    ply = a[:, 107].astype(np.int64) | (a[:, 108].astype(np.int64) << 8)
    keep = (flags & 1) == 0
    if not keep_captures:
        keep &= (flags & 2) == 0
    keep &= ply >= min_ply
    score = a[:, 103:105].copy().view("<i2")[:, 0]
    keep &= np.abs(score) < 20000
    # Malformed records (seen once: a page-cache page corrupted in RAM) would index outside the feature table.
    board = a[:, :90]
    urn = a[:, 91:103].view(np.int8)
    keep &= ~(((board & 15) > 8) | ((board >> 4) > 1)).any(1) & (a[:, 90] <= 1)
    keep &= ~((urn < 0) | (urn > np.array(URN_INIT * 2, dtype=np.int8))).any(1)
    return keep


def load_records(paths, max_records=0, keep_captures=False, min_ply=0):
    """Loads and filters record files into one preallocated array (peak memory ~ kept size), in order.
    Returns (records, total records read)."""
    chunk = 4_000_000
    counts, total = [], 0
    for p in paths:  # pass 1: kept counts per file
        m = np.memmap(p, dtype=np.uint8, mode="r")
        m = m[: len(m) // 112 * 112].reshape(-1, 112)
        total += len(m)
        counts.append(sum(int(keep_mask(np.asarray(m[i:i + chunk]), keep_captures, min_ply).sum())
                          for i in range(0, len(m), chunk)))
    out = np.empty((sum(counts), 112), dtype=np.uint8)
    pos = 0
    for p in paths:  # pass 2: fill
        m = np.memmap(p, dtype=np.uint8, mode="r")
        m = m[: len(m) // 112 * 112].reshape(-1, 112)
        for i in range(0, len(m), chunk):
            a = np.asarray(m[i:i + chunk])
            k = a[keep_mask(a, keep_captures, min_ply)]
            out[pos:pos + len(k)] = k
            pos += len(k)
    assert pos == len(out)
    if max_records and len(out) > max_records:
        out = out[-max_records:]
    return out, total


def fields(raw):
    """raw: (B,112) uint8 tensor on device -> dict of tensors."""
    board = raw[:, :90].long()
    stm = raw[:, 90].long()
    urn = raw[:, 91:103].to(torch.int8).long().view(-1, 2, 6)
    score = (raw[:, 103].long() | (raw[:, 104].long() << 8))
    score = torch.where(score >= 32768, score - 65536, score)
    result = raw[:, 105].to(torch.int8).long()
    flags = raw[:, 106].long()
    return board, stm, urn, score, result, flags


def fit_k(score, result):
    """Fit P(win) ~ sigmoid(score/K) on decisive-or-draw outcomes (score in cp, |score| < 3000)."""
    m = score.abs() < 3000
    s = score[m].double()
    y = (result[m].double() + 1) / 2
    best = None
    for K in range(100, 1500, 10):
        p = torch.sigmoid(s / K).clamp(1e-6, 1 - 1e-6)
        ll = -(y * p.log() + (1 - y) * (1 - p).log()).mean().item()
        if best is None or ll < best[1]:
            best = (K, ll)
    return best[0]

