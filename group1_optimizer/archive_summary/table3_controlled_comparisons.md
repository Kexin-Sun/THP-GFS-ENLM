# Table 3: Controlled comparisons

| Control | Configuration | Time (s) | E_common_max |
|---|---|---:|---:|
| Streaming | FJ+LS, 0 -> S+LS, 0 | 1613.3 -> 1542.4 | $2.56 \times 10^{-4}$ -> $3.26 \times 10^{-4}$ |
| Solver | S+LS, 0 -> S+EIGH, 0 | 1542.4 -> 802.7 | $3.26 \times 10^{-4}$ -> $5.64 \times 10^{-4}$ |
| Damping | S+EIGH, 0 -> S+EIGH, 1e-7 | 802.7 -> 800.5 | $5.64 \times 10^{-4}$ -> $7.71 \times 10^{-5}$ |
| Streamed, damped | FJ+EIGH, 1e-7 -> S+EIGH, 1e-7 | 880.9 -> 800.5 | $7.77 \times 10^{-5}$ -> $7.71 \times 10^{-5}$ |
