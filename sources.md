# Source Log

Date of retrieval: 2026-10-20. Every row states what was **actually retrieved and read**.
"Read" means the content was returned by a tool in this project and inspected; nothing
below is from memory unless explicitly labelled **[not re-verified]**.

Reference clones were made into a scratch directory outside this repository
(`<scratch>/refs/`). They are provenance material only; none of their code is copied
into `src/`. Commit hashes pin what was read.

## 1. Retrieval status summary

| # | Source | Method | Status |
|---|--------|--------|--------|
| S1 | ModernTCN paper (ICLR 2024) | `web_fetch` of the official proceedings PDF (text extraction) | **Full text read** (main text, Appx A-M incl. Tables 6, 11-18, 22-23, 27) |
| S2 | ModernTCN official code, `luodhhh/ModernTCN` @ `56a9a2c0` | `git clone` | **Read**: model, layer, data loader, factory, trainer, run.py args, ETTh1/ETTm1 scripts |
| S3 | ModernTCN Revisited (TMLR 2025), OpenReview `R20kKdWmVZ` | forum page blocked (bot challenge); camera-ready PDF link refused by fetch tool; **earlier submission version retrieved** via an OpenReview attachment (`id=unuOKrcs38`) found by search; abstract/bib via mlanthology and jmlr.org | **Submission version read; camera-ready NOT read** |
| S4 | Audit code, `onderakacik/RE-ModernTCN` @ `1db5eddc` | `git clone`, `diff` against S2 | **Read**: forecasting-code diff, ERF routine (`run.py`), `erf/analyze_erf.py` header |
| S5 | ETDataset, `zhouhaoyi/ETDataset` @ `1d16c8f4` | `git clone`; CSVs audited by `scripts/audit_data.py` | **Read and numerically audited** |
| S6 | Time-Series-Library, `thuml/Time-Series-Library` @ `4e938a17` (2026-04-18) | `git clone` | **Read**: `data_factory.py`, ETT loaders (borders, scaler, drop_last) |
| S7 | DLinear code, `honeywell21/DLinear` @ `f1b5359e` | `git clone` | **Read**: `models/DLinear.py`, ETT scripts, loader settings. Provenance of this fork vs. `cure-lab/LTSF-Linear` **not verified** |
| S8 | DLinear paper, arXiv 2205.13504 (AAAI 2023) | `web_fetch`/`web_search` | **Abstract/landing pages only. Paper body and tables NOT retrieved** |
| S9 | TCN code, `locuslab/TCN` @ `2f8c2b81` | `git clone` | **Read**: `TCN/tcn.py` |
| S10 | TCN paper, arXiv 1803.01271 | `web_fetch` | **Abstract only (PDF request redirected to abstract). Paper body NOT retrieved** |

Not available to this project (stated project constraint): the Lab 4 TCN implementation and
the Lab 4 autograd ERF procedure. See "Methodological adaptation" in `research_protocol.md`.

## 2. S1 - ModernTCN (ICLR 2024): verified content and implications

Locations are as printed in the retrieved PDF.

- **Block design (Sec. 3.1-3.3, Fig. 2, App. G.2, Table 19).** Input `Z in R^{M x D x N}`; feature and variable
  dims are merged so conv channels = M*D. DWConv: `groups = M*D`, large kernel, mixes time only.
  ConvFFN1: two grouped pointwise convs with `groups = M` (per-variable feature mixing).
  ConvFFN2: two grouped pointwise convs with `groups = D` after a permute (cross-variable mixing per feature).
  Hidden width of both FFNs is `r` times the channel width. Residual: `Z_{i+1} = Block(Z_i) + Z_i` (Eq. 3).
  -> Implemented as specified in `research_protocol.md` Sec. 4.
- **Patch embedding (Sec. 3.2, App. B.1).** Variable-independent: `X in R^{M x 1 x L}` -> pad by repeating the last
  value `P - S` times -> Conv1d stem (kernel P, stride S, 1 -> D channels). `N = L // S`.
  With L=336, P=8, S=4: N = 84 tokens.
- **Head (App. B.2, Eq. 5).** `Flatten(Z)` to `M x (D*N)` then a linear head to `M x T`. RevIN: per-variable
  normalise before patching, de-normalise the prediction.
- **Structural re-parameterisation (App. F.1).** Training uses a large-kernel DWConv+BN branch plus a parallel
  small-kernel DWConv+BN branch; after training they are fused. Fused model is "totally equivalent".
  -> The two branches are kept **unmerged** during training/evaluation (mathematically equivalent in eval mode).
- **Defaults (App. C.1, "Model Parameter").** 1 block, D=64, r=8, large=51, small=5, P=8, S=4; ETTm1/ETTm2 use 3 blocks;
  for small datasets such as ETTh1 "we recommend a small FFN ratio r = 1". Training: L2 loss, Adam, lr 1e-4, 100 epochs with
  early stopping, **5 seeds**, A100 40GB.
- **Splits/normalisation (App. A.1).** "split ... in chronological order by the ratio of 6:2:2 for the ETT dataset ... zero-mean
  normalized with the mean and standard deviation of training set." Table 6 lists ETTh1/ETTm1 sizes as 17,420 / 69,680.
  **Observation (data audit, S5):** the benchmark uses only the first 14,400 / 57,600 rows (12/4/4 months); 6:2:2 holds for that
  subset, not for the full file.
- **ERF claims and method (Sec. 5.2, Fig. 1, App. I).** Theory cited from Luo et al. (2016): ERF "is proportion to
  O(ks x sqrt(nl))". Visualisation: "we sample 50 length-336 input time series from the validation set in ETTh1 ... visualize how many
  points in the input series can make contribution to the middle point of the final feature map", following Kim et al. (2023).
  App. I: ModernTCN's ERF is "concentrated at the middle point and continuously expand[s] to both ends".
  **The official repository contains no ERF code** (only a README mention; checked by `grep`). The method is therefore
  reconstructed from the paper text and from S4.
- **Kernel-size table (Table 16, App. F.2.1)** - ETTh1 MSE (horizons 96/192/336/720), `D=64, K=1, r=1`:
  - k=3: 0.381 / 0.416 / 0.403 / 0.460 ; MAE 0.405 / 0.423 / 0.419 / 0.470
  - k=31: 0.367 / 0.405 / 0.389 / 0.449 ; MAE 0.393 / 0.413 / 0.410 / 0.460
  - k=51: 0.368 / 0.405 / 0.391 / 0.450 ; MAE 0.394 / 0.413 / 0.412 / 0.461
  Text (F.2.1): "Results on Table 16 show that increasing the kernel size leads to performance improvement."
  The paper does not state, for Table 16, the small-kernel size used with k=3, nor the input length.
- **Small kernel (Table 17, F.2.2).** Performance "robust to the choice of small kernel sizes as long as they're much smaller than the large kernel"; 5 recommended.
- **RevIN ablation (App. L, Table 22).** ETTh1 without RevIN: MSE 0.381 / 0.418 / 0.413 / 0.487 (96/192/336/720) vs. with RevIN 0.368 / 0.405 / 0.391 / 0.450.
- **Reported comparison values (Table 27, input length searched per model):** ModernTCN ETTh1 MSE 0.368 (H=96), 0.391 (H=336); ETTm1 0.292 (H=96), 0.365 (H=336);
  DLinear ETTh1 0.375, 0.439; ETTm1 0.299, 0.369. **Not directly comparable** to the fixed L=336 runs of this study (baselines were re-run with the best input length).
- **Implications:** (i) primary claim candidate = Table 16 / F.2.1 (see protocol Sec. 7); (ii) the head is a full-window linear map, which
  matters for ERF interpretation (protocol Sec. 6); (iii) default ETTh1 config (K=1, r=1) and ETTm1 config (K=3, r=8) are used.

## 3. S2 - Official ModernTCN code (reference only): verified facts

Files read: `ModernTCN-Long-term-forecasting/{run.py, models/ModernTCN.py, models/ModernTCN_Layer.py, data_provider/data_loader.py,
data_provider/data_factory.py, exp/exp_ModernTCN.py, utils/tools.py, layers/RevIN.py, scripts/ETTh1.sh, scripts/ETTm1.sh}`.

- **ETT script flags (verbatim from `scripts/ETTh1.sh`, `ETTm1.sh`).** `--seq_len 336`, `--patch_size 8 --patch_stride 4`, `--large_size 51 --small_size 5`,
  `--dims 64`, `--dropout 0.3`, `--batch_size 512`, `--patience 20`, `--learning_rate 0.0001`, `--train_epochs 100`, `--lradj type3`, `--itr 1`,
  `--use_multi_scale False --small_kernel_merged False`.
  ETTh1: `--ffn_ratio 1 --num_blocks 1 --head_dropout 0.0` for H=96 and 336.
  ETTm1: `--ffn_ratio 8 --num_blocks 3`; `--head_dropout 0.0` (H=96) and `0.1` (H=336). ETTm1 H=192 uses seq_len 192 and H=720 uses 672 (outside the scope).
- **Borders (`Dataset_ETT_hour`/`_minute`).** Train rows [0, 12*30*24), val starts `12*30*24 - seq_len`, test starts `12*30*24 + 4*30*24 - seq_len`; x4 for minute data.
  `StandardScaler` fit on training rows only. Time features are computed but **not used by the model** (the `te` argument is ignored in `forward_feature`).
- **`drop_last=True` for the test loader** (batch size 512). **Test loss is computed every epoch** next to validation loss and printed (it is not used for early stopping).
- **RevIN:** `affine` defaults to 0 in `run.py` (scripts do not set it); mean/std are `.detach()`-ed.
- **LR schedule `type3`:** (1-based epoch e) lr = base for e < 3, else base*0.9^(e-3) (re-read from `utils/tools.py`; DLinear's `type1` is base*0.5^(e-1)).
- **Seeds:** scripts use `--itr 1` (single run) although the paper states 5 seeds.
- **Block internals actually used:** BN after the DW branches, BN over the D feature channels before ConvFFN, GELU + dropout inside ConvFFN, residual around the whole block.

## 4. S3/S4 - ModernTCN Revisited and its code: verified content and applicability

What was read: the submission-version PDF (OpenReview attachment) and the code diff. Final TMLR title: "ModernTCN Revisited: A Critical Look at the Experimental
Setup in General Time Series Analysis"; the submission version carried a different title ("A Reproducibility Study with Extended Benchmarks ..."). Differences between the
versions were **not** checked. Numerical effect sizes quoted by the paper **[not re-verified in this session - do not quote in the report until re-checked]**.

**Verified by diff (S4 vs S2, forecasting code):** the fix applied to long-term forecasting is `drop_last=False` for the test loader
(`data_provider/data_factory.py`), plus ERF-visualisation additions to `run.py`; the ETT scripts use the same hyperparameters with `--itr 5`.

**ERF routine (S4 `run.py`, lines ~155-230), verified:** for each validation batch, `x` is permuted to `[B, M, L]`, `requires_grad=True`;
`features = model.forward_feature(x, block_idx)`; `center_idx = features.size(-1)//2`; scalar `= ReLU(features[..., center_idx]).sum()`;
`grad = autograd.grad(scalar, x)`; `grad = ReLU(grad)`; `scores = grad.sum((0,1))` (length-L vector); average over samples; saved as `.npy`. The loop counts DataLoader iterations
(equals "50 samples" only if the batch size is 1 - **not verified**). The visualisation header states it is adapted from RepLKNet's ERF script.

| Audit concern | Applies to this scope? | Reason / action |
|---|---|---|
| `drop_last=True` silently discards test windows | **Yes** | Data audit: ETTh1 loses 8.08 % (H=96) / 19.53 % (H=336) of test windows at batch 512; ETTm1 1.41 % / 3.87 %. All windows are evaluated, `drop_last=False` everywhere. |
| Test set used in validation / early stopping (M4, UEA) | **No (ETT)** | ETT has a separate validation split. The test split is additionally never evaluated before the frozen final phase. |
| Test loss printed every epoch in reference trainer | **Yes (process)** | Only validation is logged during training; test evaluated once at the end. |
| Point-adjustment in anomaly detection, classification protocol | **No** | Out of scope (forecasting only). |
| ERF comparisons against global-convolution baselines | **Partly** | Only two models are compared with one protocol; the ERF routine and its caveats are adopted (protocol Sec. 6). |
| Number of seeds (1 in scripts vs. 5 in paper) | **Yes** | 3 predefined seeds are used and mean +/- sd is reported. |

## 5. S5 - ETDataset: verified (numerical audit, `logs/data_audit.json`)

- ETTh1: 17,420 rows, ETTm1: 69,680 rows, 7 numeric columns + `date`, no NaN, no duplicate timestamps, monotonic, regular sampling (1 h / 15 min).
  Columns: `HUFL, HULL, MUFL, MULL, LUFL, LULL, OT`. (The ETDataset README's "70,080" is an approximation; the file has 69,680 rows.)
- ETTh1 is **exactly** the every-4th-row subsample of ETTm1 (identical timestamps, max |value difference| = 0.0).
- Benchmark rows used: 14,400 (ETTh1) / 57,600 (ETTm1), i.e. 82.7 % of the file; 3,020 / 12,080 trailing rows are unused by the standard protocol.
- Test period (both files): 2017-10-24 .. 2018-02-20. Train 2016-07-01 .. 2017-06-25 (last train row 2017-06-25 23:00), val targets 2017-06-26 .. 2017-10-23.
- Window counts (L=336): see `logs/data_audit.json` (e.g. ETTh1 H=96: train 8,209 / val 2,785 / test 2,785).
- Train-only standardisation (ddof=0): test OT mean is -1.34 train-sd and test OT sd is 0.34x the train sd (ETTh1). Descriptive only; its role in any failure is a hypothesis.

## 6. S6 - Time-Series-Library (current master): verified

Same borders as S2 (12/4/4 months, `border1 = start - seq_len` for val/test). `StandardScaler` fit on the train slice. `drop_last=False` for all loader flags in the version read.
Implication: the protocol of this study matches the library's current convention and the audit's corrected convention. Historical (2021-2024) versions are not reviewed here.

## 7. S7 - DLinear code (reference only): verified

`DLinear`: moving-average series decomposition (kernel 25, replicate padding), two `Linear(L -> H)` heads (trend, seasonal), summed; `individual=False` by default (shared across variables).
ETT scripts: `seq_len 336`, `batch_size 32`, lr 0.005 (ETTh1 script); `run_longExp.py` defaults 10 epochs, patience 3, `lradj type1`; test loader `drop_last=False`. No normalisation layer inside the model (data are standardised by the loader).
**Limitation:** the DLinear paper body was not retrieved (S8), so statements about the paper's claims are not made here.

## 8. S9 - TCN code (reference only): verified

`locuslab/TCN/TCN/tcn.py`: `Chomp1d` causal trimming, `TemporalBlock` = 2 x (weight-normed `Conv1d`, `Chomp1d`, ReLU, `Dropout2d`), 1x1 downsample on the residual when channel counts differ,
dilation `2^i` per level, `padding=(k-1)*d`. Receptive field with two convs per level: `1 + 2(k-1)(2^n - 1)`. The paper body (S10) was not retrieved; this formula follows from the retrieved code.

## 9. Retrieval limitations (explicit)

1. S3 camera-ready PDF and the OpenReview forum page were not retrievable (bot challenge / URL not accepted by the fetch tool); the earlier submission version was used.
2. S8 and S10 paper bodies not retrieved (arXiv returned abstracts). No claim from those papers' tables is used.
3. `honeywell21/DLinear` provenance relative to the official `cure-lab/LTSF-Linear` was not verified.
4. The Lab 4 TCN and ERF procedure are unavailable (stated project constraint); no attempt was made to locate them.
5. Official-loader window counts were derived by formula (and checked by hand for the val split), not by executing the official loader.

## 10. Files of the authors' implementation read (for the report's required list)

`luodhhh/ModernTCN`: `run.py`, `models/ModernTCN.py`, `layers/RevIN.py`, `data_provider/data_loader.py`, `data_provider/data_factory.py`, `exp/exp_ModernTCN.py`, `utils/tools.py`, `scripts/ETTh1.sh`, `scripts/ETTm1.sh`.
`onderakacik/RE-ModernTCN`: `ModernTCN-Long-term-forecasting/run.py`, `.../data_provider/data_factory.py`, `.../exp/exp_ModernTCN.py` (diffs), `.../erf/analyze_erf.py` (header).
`honeywell21/DLinear`: `models/DLinear.py`, `scripts/EXP-LongForecasting/Linear/etth1.sh`, `ettm1.sh`, `run_longExp.py`, `data_provider/data_factory.py`.
`locuslab/TCN`: `TCN/tcn.py`. `thuml/Time-Series-Library`: `data_provider/data_factory.py`, `data_provider/data_loader.py`.
