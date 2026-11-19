# Ternary KV PTQ round 2 (screening, golden fp32 + fake-quant)

Windows 0-3 of each eval set (4 x 512 tokens); dPPL vs golden fp32-KV on the same windows; no CI (4 windows).
Round-1 rows (S3, S3+s4r32) are recomputed on windows 0-3 from the 2026-11-19 score files. Generated from score files under kv_ptq2/.

| scheme | description | bits @512 | bits asym | EN dPPL | UK dPPL |
|---|---|---|---|---|---|
| S10 | LS-optimal ternary g=32 + Hadamard | 2.500 | 2.500 | +20.8% | +86.3% |
| S3 | S2 + randomized Hadamard (D=128) | 2.500 | 2.500 | +23.3% | +89.8% |
| S12 | K: Hadamard + per-channel LS-opt ternary over 32-token groups; V: S10 | 2.585 | 2.500 | +22.3% | +51.3% |
| S10+s4r32 | LS-optimal ternary g=32 + Hadamard + sinks 4 + recent 32 INT8 | 2.896 | 2.500 | +9.9% | +17.0% |
| S3+s4r32 | S2 + randomized Hadamard (D=128) + sinks 4 + recent 32 INT8 | 2.896 | 2.500 | +9.9% | +15.7% |
| S12+s4r32 | K: Hadamard + per-channel LS-opt ternary over 32-token groups; V: S10 + sinks 4 + recent 32 INT8 | 2.981 | 2.500 | +17.4% | +28.0% |
| S11 | LS-optimal ternary g=16 + Hadamard | 3.000 | 3.000 | +17.2% | +82.7% |
| S13 | K INT4 asym g=32, V S10 | 3.750 | 3.750 | +3.3% | +31.4% |
| S13+s4r32 | K INT4 asym g=32, V S10 + sinks 4 + recent 32 INT8 | 4.058 | 3.750 | +1.7% | +7.0% |
| S14 | 2-term residual ternary g=32 + Hadamard (comparator) | 5.000 | 5.000 | +1.1% | +6.8% |

## Findings (screening only)
- LS-optimal ternary threshold (S10 vs S3): small gain (EN +23.3% -> +20.8%; UK +89.8% -> +86.3%).
- g=16 (S11, 3 bits): EN +17.2%, UK +82.7%; more scales barely help UK.
- Hadamard + per-channel K over token groups (S12): no gain on EN (+22.3%), clearly better on UK (+51.3% vs +86-90%);
  with windows it is worse than S10 (+17.4% EN / +28.0% UK vs +9.9% / +17.0%).
- Ternary V is the main UK problem: K INT4 + V ternary (S13) is +3.3% EN but +31.4% UK; with sinks4+recent32 INT8
  it is +1.7% EN / +7.0% UK at ~4 bits/element.
- INT8 sink/recent windows are the dominant lever on UK (S10: +86% -> +17%).
- Two-term residual ternary (S14, 5 bits) still costs +6.8% on UK.
- No <=2.5-bit scheme below +9.9% EN / +17% UK. Finalists on full sets: not run (Part 2 took over the plan).
