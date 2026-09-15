"""Build a simulable, signed sub-connectome from the MaleCNS v1.0 release.

We do not invent connectivity.  Every edge in the resulting matrix is a measured
segment-to-segment connection from the proofread `traced-only` release table, and
every sign comes from that neuron's predicted neurotransmitter.

The full CNS is ~166k neurons / ~125M synapses.  Simulating that for hundreds of
agents in parallel is not tractable on a CPU, so we keep the circuits that
actually matter for a behaving animal and drop the rest:

  visual_projection   optic-lobe -> central-brain feature channels (LC/LPLC/LT...)
  Kenyon_Cell         mushroom body: sparse sensory code, site of learning
  MBON / DAN          mushroom body output + dopaminergic teaching signals
  CX                  central complex: heading, steering, action selection
  descending_neuron   the ~1.3k neurons that carry commands to the body

That subgraph is ~18k neurons and ~344k connections at the usual >=5-synapse
significance threshold - a real circuit, small enough to run 100+ copies at once.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import scipy.sparse as sp

from . import download

# Fly synaptic sign conventions.  ACh is the main excitatory transmitter; GABA is
# inhibitory; glutamate is predominantly inhibitory in Drosophila (GluCl-alpha);
# histamine is inhibitory (photoreceptor output).  Aminergic neurons are
# modulatory - they carry teaching signals rather than fast drive, so they get no
# fast-synaptic sign here and are handled separately by the plasticity rule.
NT_SIGN = {
    "acetylcholine": +1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
    "dopamine": 0.0,
    "octopamine": 0.0,
    "serotonin": 0.0,
}
MODULATORY = {"dopamine", "octopamine", "serotonin"}

# Populations kept in the simulable core, in the order they are laid out.
POPULATIONS = ("VPN", "KC", "MBON", "DAN", "CX", "DN", "OTHER")

CACHE = Path(__file__).resolve().parent.parent / "data" / "core_connectome.npz"


@dataclass
class Connectome:
    """A signed, simulable wiring diagram."""

    W: sp.csr_matrix          # (N, N) signed fast-synaptic weights, W[post, pre]
    Wmod: sp.csr_matrix       # (N, N) unsigned aminergic (teaching) weights
    body_ids: np.ndarray      # (N,) MaleCNS bodyId per row
    pop: np.ndarray           # (N,) index into POPULATIONS
    types: np.ndarray         # (N,) cell type string ('' when unnamed)
    nt: np.ndarray            # (N,) consensus neurotransmitter string
    side: np.ndarray          # (N,) 'L' / 'R' / ''

    @property
    def n(self) -> int:
        return self.W.shape[0]

    def idx(self, population: str) -> np.ndarray:
        """Row indices of a population, e.g. idx('KC')."""
        return np.flatnonzero(self.pop == POPULATIONS.index(population))

    def type_idx(self, *prefixes: str, population: str | None = None) -> np.ndarray:
        """Row indices whose cell type starts with any of `prefixes`."""
        mask = np.zeros(self.n, dtype=bool)
        for p in prefixes:
            mask |= np.char.startswith(self.types, p)
        if population is not None:
            mask &= self.pop == POPULATIONS.index(population)
        return np.flatnonzero(mask)

    def summary(self) -> str:
        lines = [f"MaleCNS core: {self.n} neurons, {self.W.nnz} connections"]
        for i, name in enumerate(POPULATIONS):
            k = int((self.pop == i).sum())
            if k:
                lines.append(f"  {name:6s} {k:6d}")
        exc = int((self.W.data > 0).sum())
        lines.append(f"  excitatory {exc} / inhibitory {self.W.nnz - exc}")
        lines.append(f"  modulatory (aminergic) {self.Wmod.nnz}")
        return "\n".join(lines)


def _assign_population(row_class: str, superclass: str) -> str:
    if row_class == "Kenyon_Cell":
        return "KC"
    if row_class == "MBON":
        return "MBON"
    if row_class == "DAN":
        return "DAN"
    if row_class == "CX":
        return "CX"
    if superclass == "descending_neuron":
        return "DN"
    if superclass == "visual_projection":
        return "VPN"
    return "OTHER"


def build(min_weight: int = 5, cache: Path | None = CACHE, rebuild: bool = False) -> Connectome:
    """Load (or rebuild and cache) the signed core connectome."""
    if cache is not None and cache.exists() and not rebuild:
        return _load_cache(cache)

    ann = feather.read_table(
        download.fetch("annotations"),
        columns=["bodyId", "superclass", "class", "type", "somaSide", "status"],
    ).to_pandas()
    ann = ann[ann["status"] == "Traced"]
    ann["type"] = ann["type"].fillna("")
    ann["class"] = ann["class"].fillna("")
    ann["superclass"] = ann["superclass"].fillna("")
    ann["somaSide"] = ann["somaSide"].fillna("")

    pops = np.array(
        [_assign_population(c, s) for c, s in zip(ann["class"], ann["superclass"])]
    )
    keep = pops != "OTHER"
    ann, pops = ann[keep], pops[keep]

    nt_tbl = feather.read_table(
        download.fetch("neurotransmitters"), columns=["body", "consensus_nt"]
    ).to_pandas()
    nt_map = dict(zip(nt_tbl["body"].to_numpy(), nt_tbl["consensus_nt"].to_numpy()))

    body_ids = ann["bodyId"].to_numpy()
    order = np.argsort([POPULATIONS.index(p) for p in pops], kind="stable")
    body_ids, pops = body_ids[order], pops[order]
    ann = ann.iloc[order]

    nt = np.array([str(nt_map.get(b, "unclear")) for b in body_ids])
    index = {int(b): i for i, b in enumerate(body_ids)}

    edges = feather.read_table(
        download.fetch("weights"), columns=["body_pre", "body_post", "weight"]
    ).to_pandas()
    edges = edges[edges["weight"] >= min_weight]
    pre = edges["body_pre"].to_numpy()
    post = edges["body_post"].to_numpy()
    wgt = edges["weight"].to_numpy().astype(np.float32)

    keep_e = np.fromiter((p in index and q in index for p, q in zip(pre, post)),
                         dtype=bool, count=len(pre))
    pre, post, wgt = pre[keep_e], post[keep_e], wgt[keep_e]
    ri = np.array([index[int(b)] for b in post], dtype=np.int32)
    ci = np.array([index[int(b)] for b in pre], dtype=np.int32)

    # Sign each connection by its presynaptic neuron's transmitter.  Unresolved
    # transmitters default to the population's dominant sign (KCs and VPNs are
    # overwhelmingly cholinergic); modulatory neurons contribute no fast drive.
    sign = np.array([NT_SIGN.get(t, np.nan) for t in nt], dtype=np.float32)
    unclear = np.isnan(sign)
    sign[unclear] = 1.0
    sign[unclear & (pops == "MBON")] = 0.0   # mixed; learned below, not assumed
    e_sign = sign[ci]

    # log-compress synapse counts: a 400-synapse connection is stronger than a
    # 5-synapse one, but not 80x stronger in terms of drive.
    strength = np.log1p(wgt)

    shape = (len(body_ids),) * 2
    live = e_sign != 0.0
    W = sp.csr_matrix(
        ((strength[live] * e_sign[live]).astype(np.float32), (ri[live], ci[live])),
        shape=shape,
    )
    # Aminergic output is kept unsigned and separate: DANs do not drive their
    # targets, they gate plasticity at the synapses they overlap with.
    mod = np.isin(nt[ci], list(MODULATORY))
    Wmod = sp.csr_matrix(
        (strength[mod].astype(np.float32), (ri[mod], ci[mod])), shape=shape
    )
    conn = Connectome(
        W=W, Wmod=Wmod,
        body_ids=body_ids,
        pop=np.array([POPULATIONS.index(p) for p in pops], dtype=np.int8),
        types=ann["type"].to_numpy().astype(str),
        nt=nt,
        side=ann["somaSide"].to_numpy().astype(str),
    )
    if cache is not None:
        _save_cache(conn, cache)
    return conn


def _save_cache(c: Connectome, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        data=c.W.data, indices=c.W.indices, indptr=c.W.indptr, shape=c.W.shape,
        mdata=c.Wmod.data, mindices=c.Wmod.indices, mindptr=c.Wmod.indptr,
        body_ids=c.body_ids, pop=c.pop, types=c.types, nt=c.nt, side=c.side,
    )


def _load_cache(path: Path) -> Connectome:
    z = np.load(path, allow_pickle=False)
    shape = tuple(z["shape"])
    W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=shape)
    Wmod = sp.csr_matrix((z["mdata"], z["mindices"], z["mindptr"]), shape=shape)
    return Connectome(W=W, Wmod=Wmod, body_ids=z["body_ids"], pop=z["pop"],
                      types=z["types"].astype(str), nt=z["nt"].astype(str),
                      side=z["side"].astype(str))


if __name__ == "__main__":
    print(build(rebuild=True).summary())
