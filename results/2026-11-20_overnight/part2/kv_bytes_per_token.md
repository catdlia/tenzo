| kv mode | bytes/token (all layers) | fixed bytes (all layers) | MB at ctx 2048 | MB at ctx 8192 |
|---|---|---|---|---|
| fp32 | 153600 | 0 | 300.0 | 1200.0 |
| int8_fused | 39600 | 0 | 77.3 | 309.4 |
| tl1_fused | 10800 | 0 | 21.1 | 84.4 |
| popcount_fused | 10800 | 0 | 21.1 | 84.4 |
| ternary_rot | 14400 | 1425600 | 29.5 | 113.9 |
