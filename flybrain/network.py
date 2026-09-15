"""Run a population of connectome-constrained fly brains in parallel.

Every agent in the population shares the *same* wiring diagram - the measured
MaleCNS v1.0 connectivity, signs included.  Nothing about that graph is learned.
What differs between agents, and what learning changes, are the three things
that are genuinely free in a real animal:

  1. how sensory channels are weighted onto their receptive cell types,
  2. how descending-neuron activity is read out as body commands,
  3. the strength of Kenyon-cell -> MBON synapses, which is the mushroom body's
     actual, documented site of plasticity.

(3) also changes *within* a single life, gated by dopaminergic neurons, which is
how a real fly learns.  (1) and (2) change between generations.

Two circuits are modelled rather than read off the connectome, and are marked as
such in the code: the projection-neuron drive onto Kenyon cells (the antennal
lobe is outside our simulable core, and PN->KC connectivity is close to random
anyway) and APL's feedback inhibition (implemented as k-winners-take-all, which
is what that inhibition accomplishes).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .connectome import Connectome

# Visual projection-neuron types grouped by what they are actually tuned to.
# These assignments follow the Drosophila optic-glomerulus literature: LPLC2/LC4
# are the canonical looming detectors that drive escape, LC11/LC18 respond to
# small moving objects, LC10 subtypes track objects, LPC/LLPC carry wide-field
# optic flow, and MeTu cells feed the central complex's compass.
SENSORY_CHANNELS: dict[str, tuple[str, ...]] = {
    "looming":      ("LPLC2", "LPLC1", "LC4", "LC6"),
    "small_object": ("LC11", "LC18", "LC21", "LC26"),
    "target_track": ("LC10a", "LC10d", "LC15", "LC12"),
    "optic_flow":   ("LPC1", "LPC2", "LLPC1", "LLPC2", "LLPC3"),
    "dark_edge":    ("LC16", "LC17", "LC13", "LC9"),
    "colour_cue":   ("MeTu1", "MeTu2", "MeTu3", "MeTu4"),
    "landmark":     ("LT", "LPT"),
}
# The heading channel drives the central complex compass directly, as ring
# neurons do.
COMPASS_TYPES = ("EPG", "ER", "PEN")

# Descending neurons with known behavioural roles get their own readout groups so
# the learned motor map stays interpretable; everything else is pooled by type.
NAMED_DN = ("DNa01", "DNa02", "DNp09", "DNp01", "MDN", "DNg100", "DNb01", "DNp07")

ACTIONS = ("north", "south", "east", "west", "mine", "place", "craft", "attack")


@dataclass
class Genome:
    """The free parameters of one fly.  The wiring diagram is not in here."""

    w_sense: np.ndarray   # (C, D)  obs -> sensory channel drive
    b_sense: np.ndarray   # (C,)
    w_motor: np.ndarray   # (A, G)  DN group rates -> action logits
    b_motor: np.ndarray   # (A,)
    kc_gain: np.ndarray   # (1,)    strength of PN drive onto Kenyon cells
    compass_gain: np.ndarray  # (1,)  strength of the goal-bearing ring input

    @staticmethod
    def shapes(n_ch: int, obs_dim: int, n_act: int, n_grp: int):
        return [("w_sense", (n_ch, obs_dim)), ("b_sense", (n_ch,)),
                ("w_motor", (n_act, n_grp)), ("b_motor", (n_act,)),
                ("kc_gain", (1,)), ("compass_gain", (1,))]

    @staticmethod
    def size(n_ch: int, obs_dim: int, n_act: int, n_grp: int) -> int:
        return sum(int(np.prod(s)) for _, s in Genome.shapes(n_ch, obs_dim, n_act, n_grp))


class Population:
    """A batch of `n_agents` fly brains stepped together."""

    def __init__(self, conn: Connectome, n_agents: int, obs_dim: int,
                 n_dn_groups: int = 96, seed: int = 0,
                 dt: float = 0.5, tau: float = 3.0,
                 kc_sparsity: float = 0.05, learn_rate: float = 0.06,
                 recovery: float = 0.002, plasticity_every: int = 4,
                 plastic_min_syn: int = 12, recurrent_gain: float = 0.85,
                 adaptation: float = 0.4, tau_adapt: float = 25.0,
                 sensory_gain: float = 4.0, noise: float = 0.05):
        self.conn, self.P, self.obs_dim = conn, n_agents, obs_dim
        self.dt, self.tau = dt, tau
        self.adaptation, self.tau_adapt = adaptation, tau_adapt
        self.sensory_gain = sensory_gain
        # Neurons are noisy, and that noise is what makes a fly's behaviour
        # variable enough to explore in the first place.
        self.noise = noise
        self._noise_rng = np.random.default_rng(seed + 977)
        self.kc_sparsity, self.learn_rate, self.recovery = kc_sparsity, learn_rate, recovery
        # Synaptic change is slow next to firing rates, so it is integrated every
        # few ticks instead of every tick.  Same trajectory, a quarter of the cost.
        self.plasticity_every = plasticity_every
        self._t = 0
        rng = np.random.default_rng(seed)

        # Raw synapse counts leave the network in a saturated attractor where
        # sensory input cannot compete with recurrent drive.  Scaling each
        # neuron's total input to a fixed budget - the standard normalisation for
        # connectome-derived models - keeps the relative weights of that neuron's
        # inputs (the measured quantity) while putting the whole network in a
        # regime where it actually responds to its senses.
        Wn = conn.W.tocsr().copy()
        budget = np.asarray(abs(Wn).sum(axis=1)).ravel()
        budget[budget == 0] = 1.0
        self._Wn = sp.diags((recurrent_gain / budget).astype(np.float32)) @ Wn

        self.vpn = conn.idx("VPN")
        self.kc = conn.idx("KC")
        self.mbon = conn.idx("MBON")
        self.dan = conn.idx("DAN")
        self.cx = conn.idx("CX")
        self.dn = conn.idx("DN")
        self.n = conn.n
        # `build()` lays the populations out contiguously, so each one is a slice
        # and every per-population read is a view rather than a copy.
        def _slice(ix):
            assert ix.size and np.array_equal(ix, np.arange(ix[0], ix[-1] + 1))
            return slice(int(ix[0]), int(ix[-1]) + 1)
        self.sl_kc, self.sl_mbon = _slice(self.kc), _slice(self.mbon)
        self.sl_dan, self.sl_dn = _slice(self.dan), _slice(self.dn)

        # ---- sensory channels -> concrete neuron sets -----------------------
        self.channels = list(SENSORY_CHANNELS)
        rows, chan_of, scale = [], [], []
        for ci, name in enumerate(self.channels):
            members = conn.type_idx(*SENSORY_CHANNELS[name], population="VPN")
            if members.size == 0:
                continue
            rows.append(members)
            chan_of.append(np.full(members.size, ci))
            # graded receptive scaling breaks the symmetry inside a channel
            scale.append(rng.uniform(0.4, 1.0, members.size).astype(np.float32))
        self._s_rows = np.concatenate(rows)
        self._s_chan = np.concatenate(chan_of)
        self._s_scale = np.concatenate(scale)
        self.n_ch = len(self.channels)

        # The central complex holds a ring-shaped map of heading.  Give each
        # compass neuron a preferred direction around that ring, so a bearing can
        # be written into the circuit the way a real goal-direction signal is.
        self._compass = conn.type_idx(*COMPASS_TYPES, population="CX")
        self._compass_theta = (2 * np.pi * np.arange(self._compass.size)
                               / max(self._compass.size, 1)).astype(np.float32)

        # ---- PN -> Kenyon cell drive (modelled, not measured) ---------------
        # Each KC samples a handful of input dimensions, as real KCs sample a
        # handful of glomeruli with no discernible structure.
        claws = 6
        pick = rng.integers(0, obs_dim, size=(self.kc.size, claws))
        kcp = sp.csr_matrix(
            (rng.uniform(0.5, 1.5, pick.size).astype(np.float32),
             (np.repeat(np.arange(self.kc.size), claws), pick.ravel())),
            shape=(self.kc.size, obs_dim),
        )
        self._kc_proj = kcp

        # ---- plastic KC -> MBON block ---------------------------------------
        # Only the KC->MBON connections that carry real synaptic mass are made
        # plastic; the long tail of 5-10 synapse contacts stays in the static
        # matrix, so no drive is lost but the per-agent state stays small.
        blk = self._Wn[np.ix_(self.mbon, self.kc)].tocoo()
        raw = conn.W.tocsr()[np.ix_(self.mbon, self.kc)].tocoo()
        strong = np.expm1(np.abs(raw.data)) >= plastic_min_syn
        self._pl_row = blk.row[strong].astype(np.int32)   # index into self.mbon
        self._pl_col = blk.col[strong].astype(np.int32)   # index into self.kc
        self._pl_w0 = blk.data[strong].astype(np.float32)
        self.n_plastic = self._pl_w0.size
        # scatter matrix: (n_mbon, n_edges), sums each MBON's incoming edges
        self._pl_sum = sp.csr_matrix(
            (np.ones(self.n_plastic, np.float32),
             (self._pl_row, np.arange(self.n_plastic))),
            shape=(self.mbon.size, self.n_plastic),
        )
        # static W minus the plastic edges (added back with per-agent gains)
        Wstat = self._Wn.tocoo()
        plastic_keys = set(zip((self.mbon[self._pl_row]).tolist(),
                               (self.kc[self._pl_col]).tolist()))
        keep = np.fromiter(
            ((r, c) not in plastic_keys for r, c in zip(Wstat.row.tolist(), Wstat.col.tolist())),
            dtype=bool, count=Wstat.nnz)
        self.W = sp.csr_matrix(
            (Wstat.data[keep], (Wstat.row[keep], Wstat.col[keep])), shape=Wstat.shape)

        # ---- dopaminergic teaching signal -----------------------------------
        # Which DANs innervate which MBONs, straight out of the aminergic graph.
        dm = conn.Wmod.tocsr()[np.ix_(self.mbon, self.dan)].astype(np.float32)
        norm = np.asarray(dm.sum(axis=1)).ravel()
        norm[norm == 0] = 1.0
        self._dan_to_mbon = sp.diags(1.0 / norm) @ dm

        # ---- descending-neuron readout groups -------------------------------
        self._grp_mat, self.group_names = self._build_dn_groups(n_dn_groups)
        self.n_grp = self._grp_mat.shape[0]

        self.genome_size = Genome.size(self.n_ch, obs_dim, len(ACTIONS), self.n_grp)
        self.reset()

    # ------------------------------------------------------------------ setup
    def _build_dn_groups(self, n_groups: int):
        types = self.conn.types[self.dn]
        uniq, counts = np.unique(types, return_counts=True)
        uniq = uniq[uniq != ""]
        counts = np.array([int((types == u).sum()) for u in uniq])
        chosen = list(dict.fromkeys(
            [t for t in NAMED_DN if t in set(uniq)]
            + list(uniq[np.argsort(-counts)][: n_groups - len(NAMED_DN)])
        ))[: n_groups - 1]
        rows, cols, vals, names = [], [], [], []
        assigned = np.zeros(self.dn.size, dtype=bool)
        for gi, t in enumerate(chosen):
            members = np.flatnonzero(types == t)
            assigned[members] = True
            rows.append(np.full(members.size, gi))
            cols.append(members)
            vals.append(np.full(members.size, 1.0 / members.size, np.float32))
            names.append(t)
        rest = np.flatnonzero(~assigned)
        gi = len(chosen)
        if rest.size:
            rows.append(np.full(rest.size, gi))
            cols.append(rest)
            vals.append(np.full(rest.size, 1.0 / rest.size, np.float32))
            names.append("other_DN")
        M = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(len(names), self.dn.size),
        )
        return M, names

    # ------------------------------------------------------------------ state
    def reset(self, agents: np.ndarray | None = None) -> None:
        if agents is None:
            self.r = np.zeros((self.n, self.P), np.float32)
            self.g = np.ones((self.n_plastic, self.P), np.float32)
            self._edge = np.empty((self.n_plastic, self.P), np.float32)
            self._drive = np.empty((self.n, self.P), np.float32)
            self.a = np.zeros((self.n, self.P), np.float32)
            self._t = 0
        else:
            self.r[:, agents] = 0.0
            self.a[:, agents] = 0.0
            self.g[:, agents] = 1.0

    def set_genomes(self, flat: np.ndarray) -> None:
        """flat: (P, genome_size) - one parameter vector per agent."""
        assert flat.shape == (self.P, self.genome_size), flat.shape
        o = 0
        parts = {}
        for name, shape in Genome.shapes(self.n_ch, self.obs_dim, len(ACTIONS), self.n_grp):
            k = int(np.prod(shape))
            parts[name] = flat[:, o:o + k].reshape((self.P,) + shape).astype(np.float32)
            o += k
        # store transposed so the per-agent einsums stay contiguous
        self._w_sense = np.ascontiguousarray(parts["w_sense"])       # (P,C,D)
        self._b_sense = np.ascontiguousarray(parts["b_sense"]).T     # (C,P)
        self._w_motor = np.ascontiguousarray(parts["w_motor"])       # (P,A,G)
        self._b_motor = np.ascontiguousarray(parts["b_motor"]).T     # (A,P)
        self._kc_gain = np.abs(parts["kc_gain"][:, 0]) + 0.2         # (P,)
        self._compass_w = np.abs(parts["compass_gain"][:, 0]) + 0.2   # (P,)

    # ------------------------------------------------------------------- step
    @staticmethod
    def _act(x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Rectified, saturating transfer: firing rates cannot go negative and
        cannot grow without bound.  x/(1+x/rmax) is the cheap Naka-Rushton form."""
        out = np.maximum(x, 0.0, out=out)
        return np.divide(out, 1.0 + out * (1.0 / 6.0), out=out)

    def step(self, obs: np.ndarray, bearing: np.ndarray | None = None,
             learning: bool = True) -> np.ndarray:
        """obs: (D, P).  bearing: (P,) radians toward the current objective."""
        drive = self._drive
        drive[:] = 0.0

        ch = np.einsum("pcd,dp->cp", self._w_sense, obs) + self._b_sense
        np.maximum(ch, 0.0, out=ch)
        ch *= self.sensory_gain
        drive[self._s_rows] = ch[self._s_chan] * self._s_scale[:, None]
        if bearing is not None and self._compass.size:
            bump = np.maximum(np.cos(self._compass_theta[:, None] - bearing[None, :]), 0.0)
            drive[self._compass] += (bump ** 2) * (self.sensory_gain * self._compass_w)

        # antennal-lobe drive onto Kenyon cells (modelled)
        drive[self.sl_kc] += (self._kc_proj @ obs) * self._kc_gain[None, :]

        x = self.W @ self.r
        x += drive
        # spike-frequency adaptation: sustained drive fades, so the descending
        # neurons report change rather than a fixed point
        x -= self.adaptation * self.a
        if self.noise:
            x += self._noise_rng.normal(0.0, self.noise, x.shape).astype(np.float32)

        # plastic mushroom-body output
        r_kc = self.r[self.sl_kc]
        np.take(r_kc, self._pl_col, axis=0, out=self._edge)
        self._edge *= self.g
        self._edge *= self._pl_w0[:, None]
        x[self.sl_mbon] += self._pl_sum @ self._edge

        self._act(x, out=x)
        x -= self.r
        x *= (self.dt / self.tau)
        self.r += x
        self.a += (self.dt / self.tau_adapt) * (self.r - self.a)

        # APL feedback inhibition -> only the top few percent of KCs stay active
        k = max(1, int(self.kc_sparsity * self.kc.size))
        rk = self.r[self.sl_kc]
        thresh = np.partition(rk, -k, axis=0)[-k]
        rk *= rk >= thresh[None, :]

        self._t += 1
        if learning and self._t % self.plasticity_every == 0:
            self._plasticity()

        # Action selection compares descending neurons against each other, so the
        # readout sees each group's activity relative to the rest of the DN
        # population rather than its absolute rate.
        grp = self._grp_mat @ self.r[self.sl_dn]              # (G, P)
        grp -= grp.mean(axis=0, keepdims=True)
        grp /= grp.std(axis=0, keepdims=True) + 1e-4
        return np.einsum("pag,gp->ap", self._w_motor, grp) + self._b_motor

    def _plasticity(self) -> None:
        """Dopamine-gated depression of KC->MBON synapses.

        Coincident Kenyon-cell activity and dopaminergic input to the same MBON
        weakens that synapse; synapses drift back toward baseline otherwise.
        This is the sign and the site reported for Drosophila MB plasticity.
        """
        dan_in = self._dan_to_mbon @ self.r[self.sl_dan]      # (n_mbon, P)
        coincide = np.take(self.r[self.sl_kc], self._pl_col, axis=0)
        coincide *= dan_in[self._pl_row]
        coincide *= self.g
        self.g -= (self.learn_rate * self.plasticity_every) * coincide
        self.g += (self.recovery * self.plasticity_every) * (1.0 - self.g)
        np.clip(self.g, 0.0, 1.5, out=self.g)

    # ------------------------------------------------------------- inspection
    def population_rates(self) -> dict[str, np.ndarray]:
        """Mean firing rate per population, per agent - for the HUD."""
        return {name: self.r[getattr(self, attr)].mean(axis=0)
                for name, attr in (("VPN", "vpn"), ("KC", "sl_kc"), ("MBON", "sl_mbon"),
                                   ("DAN", "sl_dan"), ("CX", "cx"), ("DN", "sl_dn"))}
