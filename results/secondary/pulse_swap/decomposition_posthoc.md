# POST-HOC: gain prior vs calibration reading (validation, pages originally at 5-10 mm/mV, seed 42)

Specified after the pulse-swap and transfer results were seen; see the module docstring of `src/decompose_pulse_effect.py`. Mean change in HYP probability.

| Model | ECGs | Consistent re-render (gain prior) | Pulse swap to 2.5 (prior + calibration) | Difference (≈ calibration reading) |
|---|---:|---:|---:|---:|
| CNN-only | 1586 | +0.034 | +0.037 | +0.002 |
| CNN-LSTM | 1586 | +0.076 | +0.308 | +0.232 |
| Temporal conv | 1586 | +0.043 | +0.150 | +0.106 |
| Hybrid ViT | 1586 | +0.033 | +0.493 | +0.460 |
