# H2 — 2x2 Disparity Analysis (Language {EN, UK} × Domain {Wiki, Web})

Corpora (zero 32-gram overlap with `calib_en8.ids` / `calib_uk8.ids` / `EN16` / `UK32`):
- `en_wiki`: `wikimedia/wikipedia` (`20231101.en`, rows 0..99)
- `uk_wiki`: `wikimedia/wikipedia` (`20231101.uk`, rows 0..99)
- `en_web`: `HuggingFaceFW/fineweb-edu` (`sample-10BT`, rows 0..99)
- `uk_web`: `Goader/kobza` (`default`, rows 200..299)

## Cell Means (16 windows per cell)

| Model | Scheme | Cell | Fertility (tok/char) | Base BPC | Mean ΔBPC | Mean ΔBPB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| BitNet-b1.58-2B | tenzo_int8 | en_wiki | 0.189 | 0.7959 | -0.00009 | -0.00009 |
| BitNet-b1.58-2B | tenzo_int8 | uk_wiki | 0.316 | 1.9225 | +0.00048 | +0.00027 |
| BitNet-b1.58-2B | tenzo_int8 | en_web | 0.207 | 0.8163 | +0.00016 | +0.00016 |
| BitNet-b1.58-2B | tenzo_int8 | uk_web | 0.336 | 2.5393 | +0.00055 | +0.00030 |
| BitNet-b1.58-2B | S3_s4r32 | en_wiki | 0.189 | 0.7959 | +0.00501 | +0.00501 |
| BitNet-b1.58-2B | S3_s4r32 | uk_wiki | 0.316 | 1.9225 | +0.04340 | +0.02389 |
| BitNet-b1.58-2B | S3_s4r32 | en_web | 0.207 | 0.8163 | +0.00938 | +0.00937 |
| BitNet-b1.58-2B | S3_s4r32 | uk_web | 0.336 | 2.5393 | +0.06510 | +0.03622 |
| BitNet-b1.58-2B | S13_s4r32 | en_wiki | 0.189 | 0.7959 | +0.00248 | +0.00248 |
| BitNet-b1.58-2B | S13_s4r32 | uk_wiki | 0.316 | 1.9225 | +0.02923 | +0.01600 |
| BitNet-b1.58-2B | S13_s4r32 | en_web | 0.207 | 0.8163 | +0.00254 | +0.00253 |
| BitNet-b1.58-2B | S13_s4r32 | uk_web | 0.336 | 2.5393 | +0.04277 | +0.02382 |
| Qwen2.5-1.5B | tenzo_int8 | en_wiki | 0.191 | 0.7675 | +0.01114 | +0.01113 |
| Qwen2.5-1.5B | tenzo_int8 | uk_wiki | 0.506 | 1.2056 | +0.06453 | +0.03537 |
| Qwen2.5-1.5B | tenzo_int8 | en_web | 0.211 | 0.7662 | +0.01828 | +0.01822 |
| Qwen2.5-1.5B | tenzo_int8 | uk_web | 0.507 | 1.6296 | +0.08320 | +0.04620 |
| Qwen2.5-1.5B | S3_s4r32 | en_wiki | 0.191 | 0.7675 | +0.22043 | +0.22035 |
| Qwen2.5-1.5B | S3_s4r32 | uk_wiki | 0.506 | 1.2056 | +0.98405 | +0.53857 |
| Qwen2.5-1.5B | S3_s4r32 | en_web | 0.211 | 0.7662 | +0.21188 | +0.21141 |
| Qwen2.5-1.5B | S3_s4r32 | uk_web | 0.507 | 1.6296 | +1.01434 | +0.56306 |
| Qwen2.5-1.5B | S13_s4r32 | en_wiki | 0.191 | 0.7675 | +1.21221 | +1.21179 |
| Qwen2.5-1.5B | S13_s4r32 | uk_wiki | 0.506 | 1.2056 | +2.94949 | +1.61971 |
| Qwen2.5-1.5B | S13_s4r32 | en_web | 0.211 | 0.7662 | +1.41151 | +1.40883 |
| Qwen2.5-1.5B | S13_s4r32 | uk_web | 0.507 | 1.6296 | +2.63729 | +1.46415 |
| EuroLLM-1.7B | tenzo_int8 | en_wiki | 0.207 | 0.7619 | +0.00010 | +0.00010 |
| EuroLLM-1.7B | tenzo_int8 | uk_wiki | 0.324 | 0.8017 | +0.00002 | +0.00001 |
| EuroLLM-1.7B | tenzo_int8 | en_web | 0.231 | 0.8031 | +0.00009 | +0.00009 |
| EuroLLM-1.7B | tenzo_int8 | uk_web | 0.334 | 1.0751 | +0.00011 | +0.00006 |
| EuroLLM-1.7B | S3_s4r32 | en_wiki | 0.207 | 0.7619 | +0.00779 | +0.00779 |
| EuroLLM-1.7B | S3_s4r32 | uk_wiki | 0.324 | 0.8017 | +0.02374 | +0.01304 |
| EuroLLM-1.7B | S3_s4r32 | en_web | 0.231 | 0.8031 | +0.02082 | +0.02078 |
| EuroLLM-1.7B | S3_s4r32 | uk_web | 0.334 | 1.0751 | +0.02952 | +0.01641 |
| EuroLLM-1.7B | S13_s4r32 | en_wiki | 0.207 | 0.7619 | +0.00166 | +0.00166 |
| EuroLLM-1.7B | S13_s4r32 | uk_wiki | 0.324 | 0.8017 | +0.00733 | +0.00404 |
| EuroLLM-1.7B | S13_s4r32 | en_web | 0.231 | 0.8031 | +0.00413 | +0.00412 |
| EuroLLM-1.7B | S13_s4r32 | uk_web | 0.334 | 1.0751 | +0.00877 | +0.00488 |

## 2×2 Factorial Regressions (`ΔBPC ~ UK + Web + UK×Web [+ base_BPC]`, 10k stratified bootstrap)

| Model | Scheme | Spec | Term | Estimate | 95% CI |
| --- | --- | --- | --- | ---: | --- |
| BitNet-b1.58-2B | tenzo_int8 | unadj_bpc | UK | +0.00057 | [-0.00093, +0.00212] |
| BitNet-b1.58-2B | tenzo_int8 | unadj_bpc | Web | +0.00025 | [-0.00032, +0.00085] |
| BitNet-b1.58-2B | tenzo_int8 | unadj_bpc | UK*Web | -0.00018 | [-0.00258, +0.00215] |
| BitNet-b1.58-2B | tenzo_int8 | adj_base_bpc | UK | +0.00197 | [-0.00244, +0.00528] |
| BitNet-b1.58-2B | tenzo_int8 | adj_base_bpc | Web | +0.00027 | [-0.00031, +0.00088] |
| BitNet-b1.58-2B | tenzo_int8 | adj_base_bpc | UK*Web | +0.00056 | [-0.00198, +0.00328] |
| BitNet-b1.58-2B | tenzo_int8 | adj_base_bpc | base_BPC | -0.00124 | [-0.00391, +0.00194] |
| BitNet-b1.58-2B | tenzo_int8 | unadj_bpb | UK | +0.00036 | [-0.00053, +0.00125] |
| BitNet-b1.58-2B | tenzo_int8 | unadj_bpb | Web | +0.00025 | [-0.00032, +0.00084] |
| BitNet-b1.58-2B | tenzo_int8 | unadj_bpb | UK*Web | -0.00022 | [-0.00162, +0.00116] |
| BitNet-b1.58-2B | S3_s4r32 | unadj_bpc | UK | +0.03839 | [+0.02833, +0.04946] |
| BitNet-b1.58-2B | S3_s4r32 | unadj_bpc | Web | +0.00437 | [+0.00039, +0.00914] |
| BitNet-b1.58-2B | S3_s4r32 | unadj_bpc | UK*Web | +0.01733 | [+0.00275, +0.03136] |
| BitNet-b1.58-2B | S3_s4r32 | adj_base_bpc | UK | +0.02484 | [-0.00076, +0.07030] |
| BitNet-b1.58-2B | S3_s4r32 | adj_base_bpc | Web | +0.00413 | [+0.00022, +0.00824] |
| BitNet-b1.58-2B | S3_s4r32 | adj_base_bpc | UK*Web | +0.01015 | [-0.00675, +0.02996] |
| BitNet-b1.58-2B | S3_s4r32 | adj_base_bpc | base_BPC | +0.01203 | [-0.02106, +0.03190] |
| BitNet-b1.58-2B | S3_s4r32 | unadj_bpb | UK | +0.01888 | [+0.01311, +0.02530] |
| BitNet-b1.58-2B | S3_s4r32 | unadj_bpb | Web | +0.00435 | [+0.00037, +0.00913] |
| BitNet-b1.58-2B | S3_s4r32 | unadj_bpb | UK*Web | +0.00797 | [-0.00115, +0.01676] |
| BitNet-b1.58-2B | S13_s4r32 | unadj_bpc | UK | +0.02675 | [+0.02187, +0.03179] |
| BitNet-b1.58-2B | S13_s4r32 | unadj_bpc | Web | +0.00006 | [-0.00259, +0.00271] |
| BitNet-b1.58-2B | S13_s4r32 | unadj_bpc | UK*Web | +0.01349 | [+0.00370, +0.02355] |
| BitNet-b1.58-2B | S13_s4r32 | adj_base_bpc | UK | +0.02426 | [+0.01471, +0.03849] |
| BitNet-b1.58-2B | S13_s4r32 | adj_base_bpc | Web | +0.00002 | [-0.00264, +0.00250] |
| BitNet-b1.58-2B | S13_s4r32 | adj_base_bpc | UK*Web | +0.01217 | [-0.00011, +0.02524] |
| BitNet-b1.58-2B | S13_s4r32 | adj_base_bpc | base_BPC | +0.00221 | [-0.01032, +0.01170] |
| BitNet-b1.58-2B | S13_s4r32 | unadj_bpb | UK | +0.01352 | [+0.01070, +0.01638] |
| BitNet-b1.58-2B | S13_s4r32 | unadj_bpb | Web | +0.00005 | [-0.00259, +0.00269] |
| BitNet-b1.58-2B | S13_s4r32 | unadj_bpb | UK*Web | +0.00777 | [+0.00193, +0.01384] |
| Qwen2.5-1.5B | tenzo_int8 | unadj_bpc | UK | +0.05340 | [+0.04244, +0.06453] |
| Qwen2.5-1.5B | tenzo_int8 | unadj_bpc | Web | +0.00715 | [+0.00078, +0.01374] |
| Qwen2.5-1.5B | tenzo_int8 | unadj_bpc | UK*Web | +0.01152 | [-0.00381, +0.02715] |
| Qwen2.5-1.5B | tenzo_int8 | adj_base_bpc | UK | +0.03553 | [+0.02425, +0.05319] |
| Qwen2.5-1.5B | tenzo_int8 | adj_base_bpc | Web | +0.00720 | [+0.00175, +0.01240] |
| Qwen2.5-1.5B | tenzo_int8 | adj_base_bpc | UK*Web | -0.00582 | [-0.02494, +0.01308] |
| Qwen2.5-1.5B | tenzo_int8 | adj_base_bpc | base_BPC | +0.04077 | [+0.01020, +0.06220] |
| Qwen2.5-1.5B | tenzo_int8 | unadj_bpb | UK | +0.02424 | [+0.01766, +0.03086] |
| Qwen2.5-1.5B | tenzo_int8 | unadj_bpb | Web | +0.00709 | [+0.00075, +0.01364] |
| Qwen2.5-1.5B | tenzo_int8 | unadj_bpb | UK*Web | +0.00374 | [-0.00634, +0.01386] |
| Qwen2.5-1.5B | S3_s4r32 | unadj_bpc | UK | +0.76362 | [+0.67231, +0.85398] |
| Qwen2.5-1.5B | S3_s4r32 | unadj_bpc | Web | -0.00855 | [-0.07577, +0.05598] |
| Qwen2.5-1.5B | S3_s4r32 | unadj_bpc | UK*Web | +0.03884 | [-0.08034, +0.16117] |
| Qwen2.5-1.5B | S3_s4r32 | adj_base_bpc | UK | +0.70793 | [+0.58794, +0.85448] |
| Qwen2.5-1.5B | S3_s4r32 | adj_base_bpc | Web | -0.00837 | [-0.07257, +0.04732] |
| Qwen2.5-1.5B | S3_s4r32 | adj_base_bpc | UK*Web | -0.01525 | [-0.16862, +0.14990] |
| Qwen2.5-1.5B | S3_s4r32 | adj_base_bpc | base_BPC | +0.12715 | [-0.11234, +0.32105] |
| Qwen2.5-1.5B | S3_s4r32 | unadj_bpb | UK | +0.31821 | [+0.25482, +0.37840] |
| Qwen2.5-1.5B | S3_s4r32 | unadj_bpb | Web | -0.00894 | [-0.07603, +0.05571] |
| Qwen2.5-1.5B | S3_s4r32 | unadj_bpb | UK*Web | +0.03343 | [-0.05142, +0.11894] |
| Qwen2.5-1.5B | S13_s4r32 | unadj_bpc | UK | +1.73728 | [+1.61447, +1.87913] |
| Qwen2.5-1.5B | S13_s4r32 | unadj_bpc | Web | +0.19931 | [+0.06775, +0.33930] |
| Qwen2.5-1.5B | S13_s4r32 | unadj_bpc | UK*Web | -0.51151 | [-0.71962, -0.31231] |
| Qwen2.5-1.5B | S13_s4r32 | adj_base_bpc | UK | +1.64274 | [+1.40429, +1.92131] |
| Qwen2.5-1.5B | S13_s4r32 | adj_base_bpc | Web | +0.19960 | [+0.07070, +0.32971] |
| Qwen2.5-1.5B | S13_s4r32 | adj_base_bpc | UK*Web | -0.60334 | [-0.83291, -0.33461] |
| Qwen2.5-1.5B | S13_s4r32 | adj_base_bpc | base_BPC | +0.21585 | [-0.23617, +0.55643] |
| Qwen2.5-1.5B | S13_s4r32 | unadj_bpb | UK | +0.40792 | [+0.31885, +0.50689] |
| Qwen2.5-1.5B | S13_s4r32 | unadj_bpb | Web | +0.19704 | [+0.06604, +0.33777] |
| Qwen2.5-1.5B | S13_s4r32 | unadj_bpb | UK*Web | -0.35259 | [-0.52351, -0.18870] |
| EuroLLM-1.7B | tenzo_int8 | unadj_bpc | UK | -0.00008 | [-0.00051, +0.00034] |
| EuroLLM-1.7B | tenzo_int8 | unadj_bpc | Web | -0.00001 | [-0.00033, +0.00030] |
| EuroLLM-1.7B | tenzo_int8 | unadj_bpc | UK*Web | +0.00010 | [-0.00053, +0.00077] |
| EuroLLM-1.7B | tenzo_int8 | adj_base_bpc | UK | -0.00009 | [-0.00053, +0.00036] |
| EuroLLM-1.7B | tenzo_int8 | adj_base_bpc | Web | -0.00002 | [-0.00034, +0.00030] |
| EuroLLM-1.7B | tenzo_int8 | adj_base_bpc | UK*Web | +0.00006 | [-0.00064, +0.00083] |
| EuroLLM-1.7B | tenzo_int8 | adj_base_bpc | base_BPC | +0.00018 | [-0.00100, +0.00132] |
| EuroLLM-1.7B | tenzo_int8 | unadj_bpb | UK | -0.00009 | [-0.00037, +0.00018] |
| EuroLLM-1.7B | tenzo_int8 | unadj_bpb | Web | -0.00001 | [-0.00033, +0.00030] |
| EuroLLM-1.7B | tenzo_int8 | unadj_bpb | UK*Web | +0.00006 | [-0.00038, +0.00051] |
| EuroLLM-1.7B | S3_s4r32 | unadj_bpc | UK | +0.01595 | [+0.01206, +0.01982] |
| EuroLLM-1.7B | S3_s4r32 | unadj_bpc | Web | +0.01303 | [+0.00691, +0.02036] |
| EuroLLM-1.7B | S3_s4r32 | unadj_bpc | UK*Web | -0.00725 | [-0.01757, +0.00288] |
| EuroLLM-1.7B | S3_s4r32 | adj_base_bpc | UK | +0.01482 | [+0.00957, +0.01965] |
| EuroLLM-1.7B | S3_s4r32 | adj_base_bpc | Web | +0.01186 | [+0.00624, +0.01726] |
| EuroLLM-1.7B | S3_s4r32 | adj_base_bpc | UK*Web | -0.01386 | [-0.02568, -0.00089] |
| EuroLLM-1.7B | S3_s4r32 | adj_base_bpc | base_BPC | +0.02848 | [-0.00100, +0.05132] |
| EuroLLM-1.7B | S3_s4r32 | unadj_bpb | UK | +0.00525 | [+0.00263, +0.00781] |
| EuroLLM-1.7B | S3_s4r32 | unadj_bpb | Web | +0.01299 | [+0.00688, +0.02032] |
| EuroLLM-1.7B | S3_s4r32 | unadj_bpb | UK*Web | -0.00963 | [-0.01794, -0.00195] |
| EuroLLM-1.7B | S13_s4r32 | unadj_bpc | UK | +0.00567 | [+0.00346, +0.00790] |
| EuroLLM-1.7B | S13_s4r32 | unadj_bpc | Web | +0.00247 | [+0.00121, +0.00368] |
| EuroLLM-1.7B | S13_s4r32 | unadj_bpc | UK*Web | -0.00103 | [-0.00459, +0.00251] |
| EuroLLM-1.7B | S13_s4r32 | adj_base_bpc | UK | +0.00538 | [+0.00301, +0.00775] |
| EuroLLM-1.7B | S13_s4r32 | adj_base_bpc | Web | +0.00217 | [+0.00086, +0.00340] |
| EuroLLM-1.7B | S13_s4r32 | adj_base_bpc | UK*Web | -0.00273 | [-0.00632, +0.00104] |
| EuroLLM-1.7B | S13_s4r32 | adj_base_bpc | base_BPC | +0.00730 | [+0.00199, +0.01205] |
| EuroLLM-1.7B | S13_s4r32 | unadj_bpb | UK | +0.00237 | [+0.00103, +0.00371] |
| EuroLLM-1.7B | S13_s4r32 | unadj_bpb | Web | +0.00246 | [+0.00120, +0.00367] |
| EuroLLM-1.7B | S13_s4r32 | unadj_bpb | UK*Web | -0.00162 | [-0.00385, +0.00065] |
