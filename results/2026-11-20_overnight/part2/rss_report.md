# Engine RSS (T31)

Numbers: `rss/rss_summary.md` (RSS and HWM per KV mode and context, before/after), `rss/rss_breakdown.md` (RSS by mapping from
/proc/self/smaps after engine creation, after load, after 16 + 128 tokens). Protocol: `tenzo-inference --bench-pp 16 --bench-tg 128`,
model `models/bitnet-1.58b-v2`, 4 threads; "part-1 end" = binary built from b784698 (end of part 1).

**Peak (VmHWM) at ctx 2048, int8_fused: 2761 MB -> 935 MB; steady RSS 994 -> 935 MB; target <= 1.5 GB reached (peak == steady).**
At ctx 8192: HWM 3225 -> 1166 MB, RSS 1459 -> 1166 MB.

What made the peak (found with the smaps dumps and the load code):
1. `load_model_from_files` read the whole 1836 MB weights.bin into a `std::vector` (1.31 GB of it fp32 embeddings that are only
   quantized to the INT8 table) and kept the TL1 copy of every layer next to the new I2_S copy.
   Fix: private read-only mmap of the file (`WeightsFile`), the converted pages are dropped with `madvise(MADV_DONTNEED)`: every
   4096 embedding rows (42 MB steps) and after each layer; the I2_S rows are built per layer and the TL1 copy is freed at once.
   An intermediate build with only the mmap (no progressive release) still peaked at 1747 MB: the mapped fp32 table became resident
   while it was being quantized.
2. Steady state: `PagedKVCache` was always allocated and zero-filled although only `paged_int8` uses it (a second full INT8 KV
   allocation: ~80 MB at ctx 2048, ~315 MB at ctx 8192). It is now a 1-block stub unless kv_mode is `paged_int8`.

Composition of the final steady state at ctx 2048 (after_load): ~929 MB anonymous = INT8 embedding/LM-head table 328 MB +
I2_S weights 521 MB + KV cache (INT8, 2048 positions) ~80 MB + ~0.5 MB scratch; nothing else of significance (shared libraries 4-5 MB).
So the remaining footprint is the model itself.

Gates: G6 of all five KV modes unchanged (`logs/g6_rss.log`); PPL of int8_fused on windows 0-3 identical per window to the
2026-11-19 run (16.700700, `rss/ppl4_int8_fused.txt`).
