# Engine RSS: steady state (VmRSS) and peak (VmHWM)

Measured with `tenzo-inference --bench-pp 16 --bench-tg 128 -c <ctx>` (model models/bitnet-1.58b-v2, int8 LM head, I2_S GEMV); RSS at exit after 16 prompt + 128 generated tokens.

| kv mode | ctx | part-1 end RSS MB | part-1 end HWM MB | mmap loader RSS MB | mmap loader HWM MB | final RSS MB | final HWM MB |
|---|---|---|---|---|---|---|---|
| int8_fused | 2048 | 994 | 2761 | 1011 | 1747 | 935 | 935 |
| fp32 | 8192 | - | - | 2352 | 3101 | 2057 | 2057 |
| int8_fused | 8192 | 1459 | 3225 | 1475 | 2211 | 1166 | 1166 |
| ternary_rot | 8192 | - | - | 1280 | 2015 | 971 | 971 |
| tl1_fused | 8192 | - | - | 1250 | 1986 | 941 | 941 |
