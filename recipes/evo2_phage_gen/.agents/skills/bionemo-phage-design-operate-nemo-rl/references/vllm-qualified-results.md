# Evo2 vLLM qualification record

This is the retained two-H100 qualification record for the recipe-owned Evo2
vLLM integration. It is computational evidence, not a biological viability
claim. Re-run the same gates after changing the model export, tokenizer,
dependency lock, recipe patches, runtime profile, topology, or batch geometry.

## Runtime and workload

- Production GDPO workload: eight prompt-length strata (4 through 11), twelve
  stochastic rollouts per stratum, `P=8`, `K=12`, `GBS=96`, exact 5,988-token
  completions, full external QC, and frozen mixed validation.
- TP2/DP1 generation route: official vLLM, MP executor, async scheduling,
  O2/balanced, compilation mode 3, `FULL_AND_PIECEWISE` CUDA graphs with exact
  B96 capture, processed chosen-token logprobs, and no upstream vLLM patch.
- Source revision for the accepted TP2/DP1 run:
  `c0b057d63f23d31d73f3aa5bfd825984fd5529da`.
- Source revision for the accepted TP1/DP2 and mixed-accuracy runs:
  `45089b2619e0252765311bef0580cd77a74be804`.
- Source revision for the final clean mixed-accuracy and public-inference
  reruns: `5f4969151ca68aa0d4ed2445420796de7e9173d4`.
- The accepted TP2/DP1 run validated RL/standalone load parity for 330 tensors,
  the exported config and manifest, and tokenizer SHA256
  `52ccdc9c776e79a6c005d6b55d271e1dfaba55e60b52fe9bb2d1fec22d407504`.

## GDPO timing

The matched native comparison is the recovered early production TP2/DP1
series: 511.7 seconds for a train-only step, 840.3 seconds for a step with
validation and checkpoint time removed, 369.7 seconds for generation plus
nested reward/QC, 37.2 seconds for policy plus reference logprobs, and 66.2
seconds for policy training.

| Measurement | vLLM TP2/DP1 | Matched native | Difference |
| --- | ---: | ---: | ---: |
| Train-only step, three-step median | 385.83 s | 511.7 s | 24.60% lower |
| Step 3 including final validation | 590.48 s | 840.3 s | 29.73% lower |
| Generation plus nested reward/QC, median | 238.15 s | 369.7 s | 35.58% lower |
| Policy plus reference logprobs, median | 29.81 s | 37.2 s | 19.87% lower |
| Policy training, median | 53.20 s | 66.2 s | 19.64% lower |

TP2/DP1 train step totals were 318.86, 385.83, and 386.43 seconds; the last
value excludes its 204.05-second final validation. Model generation waves were
80.72, 133.27, and 106.72 seconds, with final validation generation at 80.63
seconds. Every optimizer step completed conversion/refit/synchronization before
the next rollout or final validation.

The independent TP1/DP2 run completed the same P8/K12/GBS96 contract with two
48-row streams. Its train-only totals were 321.73, 327.87, and 360.65 seconds,
for a 327.87-second median (35.93% below the matched native TP2/DP1 median).
Its step containing validation was 578.85 seconds (31.11% below the matched
native value). These topology results are both functional evidence; they are
not a same-topology speed comparison.

## Numerical and biological-output gates

The TP2/DP1 audit reopened 384 rows: three train waves and one validation wave,
96 rows each. All 2,299,392 completion tokens were exact-length A/C/G/T/N with
no EOS; token IDs decoded to the retained text; chosen rollout and policy
logprobs were finite and aligned. All 384 request IDs, global indices, and seeds
were unique. Generation calls advanced from 0 through 3 with disjoint seed
ranges. Rewards, advantages, losses, gradients, optimizer state, and post-refit
validation were finite.

For rollout versus training-policy logprobs, absolute token delta had mean
0.02401, p95 0.13860, and max 5.49229. The configured multiplicative sequence
error had mean 1.03076, p95 1.07897, and max 1.10170; zero rows exceeded the
unchanged 1.5 threshold. Token importance ratios had mean 1.01046 and p95
1.06242; 3.496% were outside the PPO clip interval `[0.8, 1.2]`.

The final clean canonical mixed TP2 identity gate passed unequal-prefix B4
with match counts `[490, 475, 390, 428]`. All 96 interleaved B96 occurrences
passed unchanged per-case floors `[440, 404, 361, 381]`; observed per-case
minimum match counts were `[486, 475, 390, 395]`. Every occurrence contained
exactly 500 DNA tokens and aligned finite chosen logprobs. The B96 generation
call completed 48,000 tokens in 16.450 seconds. This short accuracy workload
is not the long-generation throughput authority.

The final public `infer_evo2` persistent P8/K12 control passed 192/192
exact-5,988 rows. Its steady generation wall was 82.602 seconds, or 6,959.22
completion tokens/second. That is 21.18% above the matched native warm
5,742.8-token/second reference and within 0.18% of the accepted direct actor
control. The run used the same TP2 MP+async O2/balanced exact-B96 graph route
as GDPO and independently bound the exported model to the retained RL
checkpoint and tokenizer. This standalone number remains a control; GDPO
comparisons use the complete outer step, including reward/QC, training, and
refit.

## Required final filters

The target final-pass waterfall requires AAI (filter 8), the required-gene
list, and synteny/total-gene logic (filter 9). None is optional in final-pass
logging. Architecture-removal filter 7 remains intentionally disabled for this
target profile. Report both raw and 99%-identity-deduplicated denominators and
do not combine the online validation and offline rollout contracts.

## Retained evidence

Machine-local artifact roots are under `/data/jstjohn/evo2-vllm-lab/artifacts`:

- `gdpo-tp2dp1-p8k12-exact6k-fullqc-c0b057d6-a2/run.log`:
  `35f0eb57ca5e654b0885e4e5c63e99487566db429838d5e975b4529c5b86ca30`.
- TP2 train row SHA256 values by step:
  `2aed7992e149227730bd45d939708919919feb42c86fdfa06cd6326c93cfb174`,
  `1c4d7050d27acc0e2bf25fdc9f15b8a617d93f30fb1a87eb5a10345931968765`,
  and `3f7012b79f0bf15d63876c465e4b8109b4f1ffed6dc8b74b3b89cd865b10b05c`.
- TP2 validation rows:
  `b44f3549d1e76aa633d66955635358a4e5ccc8133d42853b713a841490a65dfa`.
- `gdpo-tp1dp2-p8k12-exact6k-fullqc-45089b26-a2/run.log`:
  `75ccbbaf6f45494af934a16e10b09791330f16d0263e1bdf33a701412a34dde2`.
- `final-tp2-mixed-b4-b96-5f496915-mp-async-a1.json`:
  `d164d2d5ef1876b99ebb873fc25d0f837c6647f75af92e385442ec7fbcb66296`.
- Final mixed B4 and B96 full-output sidecars:
  `5abded06393f3b2ec09358b2675c4638aa5136608f7fed4e8ef2a095235f90f7`
  and `5fa25eaa2e9fef729ad142a8b411ca2d964c85de2cf196667d3460dba1064be2`.
- Final public-inference manifest, rows, and log:
  `fcf9e69ab755e2740b33afa2115f3756051b07b7273860a25d5649f2256ee5c3`,
  `23461abd3c3c858544739b43baaa8da6ffb8793369d1d267acbcf337d6567cbf`,
  and `fa55f1f4ed60496d4004709f02430ef2e67ae30348444f8324c21a679e49c6a8`.

These hashes bind the retained lab evidence. A portable release should copy the
selected artifacts into its declared result root or artifact store rather than
assuming these machine-local paths exist.
