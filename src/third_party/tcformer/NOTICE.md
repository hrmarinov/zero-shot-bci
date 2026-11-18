# Vendored from https://github.com/Altaheri/TCFormer (MIT, see LICENSE)

`modules.py` and `channel_group_attention.py` are unchanged copies. `tcformer_model.py` is `models/tcformer.py`
with the PyTorch-Lightning wrapper class (`TCFormer(ClassificationModule)`), the latency benchmark import and
the `utils.weight_initialization` import removed (the Glorot initialiser is inlined). Only `TCFormerModule`
(a plain `torch.nn.Module`) is used, by `experiments/t1_tcformer_head_to_head.py`. Fetched 2026-11-18 as data
and read before use.
