"""The numpy decoders, driven by a fake decoder whose next-token scores are fixed."""

import numpy as np

from netra_nmt.inference.nmt import NMTModel, log_softmax

V, BOS, EOS = 6, 2, 3


class FakeModel(NMTModel):
    """Step t prefers TABLE[t]; the KV cache only carries the length."""

    def __init__(self, table):
        self.layers, self.heads, self.head_dim = 1, 1, 1
        self.bos_id, self.eos_id, self.pad_id = BOS, EOS, 0
        self.table = np.asarray(table, dtype=np.float32)
        self._on_gpu = False

    def _encode(self, ids, keep):
        return np.zeros((1, 1, 1, len(ids), 1), np.float32), np.zeros((1, 1, 1, len(ids), 1), np.float32), \
            np.zeros((1, len(ids)), np.float32)

    def _step(self, ids, past_k, past_v, cross_k, cross_v, cross_mask, keep):
        batch, t = len(ids), past_k.shape[3]
        logits = np.repeat(self.table[min(t, len(self.table) - 1)][None], batch, axis=0)
        grow = np.zeros(past_k.shape[:3] + (1,) + past_k.shape[4:], np.float32)
        return logits, np.concatenate([past_k, grow], 3), np.concatenate([past_v, grow], 3)


TABLE = [
    [0, 0, 0, 0, 5, 1],     # step 0: token 4
    [0, 0, 0, 0, 1, 5],     # step 1: token 5
    [0, 0, 0, 9, 0, 0],     # step 2: EOS
]


def test_log_softmax_normalises():
    x = np.array([[1.0, 2.0, 3.0]], np.float32)
    assert np.allclose(np.exp(log_softmax(x)).sum(), 1.0)


def test_greedy_stops_at_eos():
    assert FakeModel(TABLE).greedy([7, 8]) == [BOS, 4, 5, EOS]


def test_greedy_respects_max_new_tokens():
    assert FakeModel(TABLE).greedy([7], max_new_tokens=2) == [BOS, 4, 5]


def test_beam_finds_the_greedy_path_on_a_peaked_table():
    assert FakeModel(TABLE).beam([7, 8], beam_size=3) == [BOS, 4, 5, EOS]


def test_sample_is_seeded_and_ends_at_eos():
    m = FakeModel(TABLE)
    a = m.sample([7], seed=0, top_p=0.5)
    assert a == m.sample([7], seed=0, top_p=0.5)
    assert a == [BOS, 4, 5, EOS]   # top-p 0.5 keeps only the peak token each step
