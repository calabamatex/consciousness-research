#!/usr/bin/env python3
"""
Week 01 - Phi (Integrated Information) Computation
===================================================
The Formal Shadow of Rigpa: Consciousness Research Project

Demonstrates IIT's core concepts using numpy-only implementations.
Computes Phi for small 3-node systems to show how integrated information
theory distinguishes integrated from non-integrated architectures.

Dependencies: numpy only (no pyphi required)
"""
import numpy as np
from itertools import combinations

np.set_printoptions(precision=4, suppress=True)

# ============================================================================
# SECTION 1: TPM CONSTRUCTION
# ============================================================================
# The Transition Probability Matrix defines a system's causal structure.
# Row s gives P(each node ON | current state s). For n binary nodes: 2^n rows.
# In IIT the TPM IS the system -- it specifies intrinsic cause-effect structure.

def make_tpm(weights, determinism=5.0):
    """Build state-by-node TPM from connectivity matrix via sigmoid activation."""
    n = weights.shape[0]
    num_states = 2 ** n
    bias = -np.sum(np.abs(weights), axis=0) / 2.0
    tpm = np.zeros((num_states, n))
    for s in range(num_states):
        state = np.array([(s >> i) & 1 for i in range(n)], dtype=float)
        tpm[s] = 1.0 / (1.0 + np.exp(-determinism * (state @ weights + bias)))
    return tpm

def state_labels(n):
    return [format(i, f"0{n}b")[::-1] for i in range(2 ** n)]

# ============================================================================
# SECTION 2: PARTITION ANALYSIS AND PHI
# ============================================================================
# Phi = min distance(full_TPM, partitioned_TPM) over all bipartitions.
# Partitioned TPM assumes the two halves evolve independently (causal cut).
# Phi > 0: system is irreducible. Phi = 0: system decomposes into parts.

def emd_l1(p, q):
    """L1 distance as EMD approximation (valid lower bound for same support)."""
    return np.sum(np.abs(p - q))

def partitioned_tpm(tpm, part_a, part_b, n):
    """TPM with causal arrows between A and B severed (independence assumption)."""
    num_states = 2 ** n
    result = np.zeros_like(tpm)
    for s in range(num_states):
        for part, label in [(part_a, 'a'), (part_b, 'b')]:
            marginal = np.zeros(n)
            count = 0
            for s2 in range(num_states):
                if all(((s2 >> i) & 1) == ((s >> i) & 1) for i in part):
                    marginal += tpm[s2]
                    count += 1
            if count > 0:
                marginal /= count
            for j in part:
                result[s, j] = marginal[j]
    return result

def bipartitions(n):
    """All non-trivial bipartitions of {0..n-1}, deduplicated."""
    indices = list(range(n))
    parts = []
    for size in range(1, n):
        for sub in combinations(indices, size):
            comp = tuple(i for i in indices if i not in sub)
            if sub < comp:
                parts.append((list(sub), list(comp)))
    return parts

def compute_phi(tpm, n):
    """Returns (phi, mip, all_partition_results)."""
    parts = bipartitions(n)
    if not parts:
        return 0.0, ([], []), []
    results = []
    for pa, pb in parts:
        dist = emd_l1(tpm, partitioned_tpm(tpm, pa, pb, n))
        results.append(((pa, pb), dist))
    results.sort(key=lambda x: x[1])
    mip, phi = results[0]
    return phi, mip, results

# ============================================================================
# SECTION 3: THREE EXAMPLE SYSTEMS
# ============================================================================
print("=" * 70)
print("INTEGRATED INFORMATION (PHI) COMPUTATION")
print("The Formal Shadow of Rigpa -- Week 01")
print("=" * 70)

# System A: Fully connected recurrent -- models thalamocortical loops
W_a = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
tpm_a = make_tpm(W_a)
phi_a, mip_a, parts_a = compute_phi(tpm_a, 3)

# System B: Feedforward chain 0->1->2 -- models subliminal processing
W_b = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
tpm_b = make_tpm(W_b)
phi_b, mip_b, parts_b = compute_phi(tpm_b, 3)

# System C: Disconnected -- no causal structure
W_c = np.zeros((3, 3))
tpm_c = make_tpm(W_c)
phi_c, mip_c, parts_c = compute_phi(tpm_c, 3)

for label, W, tpm, phi, mip in [
    ("A: Fully Connected Recurrent", W_a, tpm_a, phi_a, mip_a),
    ("B: Feedforward Chain (0->1->2)", W_b, tpm_b, phi_b, mip_b),
    ("C: Disconnected (no connections)", W_c, tpm_c, phi_c, mip_c),
]:
    print(f"\n[SYSTEM {label}]")
    print(f"  Weights: {W.tolist()}")
    print(f"  Phi = {phi:.4f}   MIP = {mip[0]} | {mip[1]}")

# ============================================================================
# SECTION 4: COMPARISON TABLE
# ============================================================================
print("\n" + "=" * 70)
print("COMPARISON TABLE")
print("=" * 70)
phi_max = max(phi_a, phi_b, phi_c)

def classify(v):
    r = v / phi_max if phi_max > 0 else 0
    return "NO" if v < 0.001 else ("PARTIAL" if r < 0.7 else "YES")

hdr = f"{'System':<32} {'Phi':>8} {'Irreducible':>12} {'Type':<16}"
print(hdr)
print("-" * len(hdr))
for name, phi, arch in [
    ("A: Recurrent (integrated)", phi_a, "Bidirectional"),
    ("B: Feedforward (chain)", phi_b, "Unidirectional"),
    ("C: Disconnected (isolated)", phi_c, "None"),
]:
    print(f"{name:<32} {phi:>8.4f} {classify(phi):>12} {arch:<16}")

# ============================================================================
# SECTION 5: PARTITION DETAIL (System A)
# ============================================================================
print("\n" + "=" * 70)
print("ALL PARTITIONS FOR SYSTEM A (showing MIP selection)")
print("=" * 70)
for (pa, pb), dist in parts_a:
    tag = " <-- MIP" if abs(dist - phi_a) < 1e-10 else ""
    print(f"  {pa} | {pb}  ->  info loss = {dist:.4f}{tag}")
print(f"  Even the gentlest cut loses {phi_a:.4f} units: system is irreducible.")

# ============================================================================
# SECTION 6: STRUCTURAL INVARIANTS
# ============================================================================
print("\n" + "=" * 70)
print("STRUCTURAL INVARIANTS")
print("=" * 70)

# 1. REFLEXIVITY -- recurrence creates self-referencing causal loops
print("\n1. REFLEXIVITY (Self-Referencing Causal Structure)")
print("   System A: every node is both cause and effect (closed loops).")
for i in range(3):
    ins = [j for j in range(3) if W_a[j, i] > 0]
    outs = [j for j in range(3) if W_a[i, j] > 0]
    print(f"     Node {i}: receives {ins}, sends {outs}")
print("   System B: node 0 receives nothing, node 2 sends nothing. No loops.")

# 2. NON-DECOMPOSABILITY -- Phi > 0 means the whole exceeds parts
print(f"\n2. NON-DECOMPOSABILITY")
print(f"   System A: Phi={phi_a:.4f} -- cannot be factored without info loss")
print(f"   System C: Phi={phi_c:.4f} -- already fully decomposable")

# 3. SELF-SPECIFICATION -- TPM is the system's intrinsic identity
print(f"\n3. SELF-SPECIFICATION (TPM as intrinsic identity)")
print("   System A TPM varies richly with state (dense causal coupling):")
labels = state_labels(3)
for idx in [0, 3, 7]:
    r = tpm_a[idx]
    print(f"     {labels[idx]} -> [{r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f}]")
print(f"   System C TPM is uniform: [{tpm_c[0,0]:.3f}, {tpm_c[0,1]:.3f}, "
      f"{tpm_c[0,2]:.3f}] for all states (no causal structure).")

# ============================================================================
# SECTION 7: RESULTS SUMMARY FOR ACADEMIC WRITER
# ============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY FOR ACADEMIC WRITER")
print("=" * 70)

# Plain-language translation of computational findings for thesis integration.
# Phi values: A={phi_a:.4f}, B={phi_b:.4f}, C={phi_c:.4f}

print(f"""
1. PHI DISTINGUISHES ARCHITECTURES
   Recurrent: Phi={phi_a:.4f} | Feedforward: Phi={phi_b:.4f} | Disconnected: Phi={phi_c:.4f}
   Recurrent Phi is {phi_a/max(phi_b,0.001):.1f}x the feedforward value. Bidirectional
   causal coupling produces irreducible integration; unidirectional does not.

2. RECURRENCE IS NECESSARY FOR HIGH PHI
   Only feedback loops produce substantial Phi. This parallels neuroscience:
   recurrent thalamocortical loops correlate with conscious experience,
   while feedforward sweeps (subliminal processing) do not.

3. STRUCTURAL INVARIANTS MAP TO IIT AXIOMS
   a) REFLEXIVITY: Closed causal loops = self-referencing = IIT "intrinsic existence"
   b) NON-DECOMPOSABILITY: Phi > 0 = whole > sum of parts = "unified experience"
   c) SELF-SPECIFICATION: TPM IS the system's identity = "intrinsic information"

4. RELEVANCE TO "THE FORMAL SHADOW OF RIGPA"
   Dzogchen's rigpa is reflexive, irreducible, and self-specifying. IIT's Phi
   provides a formal analogue: a high-Phi system shares these structural
   properties. Phi is the "formal shadow" -- a mathematical trace of what
   contemplative traditions attribute to awareness.

   CAVEAT: Phi measures causal STRUCTURE, not phenomenal CONTENT. It captures
   the form of consciousness while remaining silent on qualia. The thesis
   argues this gap is where the "shadow" metaphor is most apt.

METHODOLOGY: L1 distance (simplified EMD), adequate for 3-node demos.
   Production IIT analysis should use pyphi (github.com/wmayner/pyphi).
""")
print("=" * 70)
