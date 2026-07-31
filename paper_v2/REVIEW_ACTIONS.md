# REVIEW_ACTIONS.md — 外部審閱清單 A–N 的逐項執行追蹤

> 來源：2026-07-29 使用者提供的 code-agent 實作清單（A–N，約 50 項）。
> 狀態碼：**DONE** 已完成並驗證｜**PARTIAL** 部分完成（註明缺口）｜**TODO** 未動｜
> **BLOCKED** 需資源或作者決定。本檔是這份清單的唯一權威追蹤表；專案整體狀態見 `STATUS.md`。
> 最後更新：2026-07-30（B3 結論撤回並更正）。

---

## A. 數學與實驗一致性（P0）— 全部完成

| 項 | 內容 | 狀態 |
|---|---|---|
| A1 | 符號與診斷量不一致（稿中 z_g=−8.3 但定義下恆正） | **DONE** |
| A2 | 不等 n_i 下的 variance identity | **DONE** |
| A3 | collapse probability 近似的 Monte Carlo 網格驗證 | **DONE** |
| A4 | 降低 theorem claim 強度 | **DONE** |
| A5 | oracle inequality 的 B 有界性與獨立單位 | **DONE** |

**A1**：使用者指出的是真錯誤，且是我引入的。稿中表格的量是用**未截斷** plug-in
ρ̂_g=(S_g²−s̄_g)/S_g² 算的**帶號**統計量；proposition 中的 ρ_g=τ_g²/(τ_g²+s̄_g) 是母體量、恆正。
處置：母體量維持 ρ_g；經驗診斷正式命名為 **ẑ_g=(S_g²−s̄_g)/ŜE(S_g²)**，
ŜE(S_g²)=√(2/(m_g−1))·S_g²，ẑ_g≤0 恰為 collapse 事件。稿中明寫「兩個符號全篇分開使用」。
新增 `scripts/analysis/resolution_diagnostics.py`，對每個 component 輸出
m_g / S_g² / s̄_g / τ̂²(MoM) / τ²(未截斷) / ρ̂ / ẑ / collapse 指標 → `outputs/aistats2027/resolution_diagnostics.csv`。

**A2**：**精確成立**，三個條件已對程式碼核對（`_estimate_size_hyper_from_stats`）：
S_g² 用 `np.var(y_i, ddof=1)`（1/(m_g−1)）；s̄_g 用 `np.mean(sigma_sq/n_i)`（**未加權**）；
items 在工作模型下獨立、sampling variance 異質。此三條件下 E[S_g²]=τ_g²+s̄_g
**不需平衡假設、不需權重修正**（sympy 符號驗證）。稿中已加限定：此恆等式是對 **MoM 估計式**
（程式中 REML 的初始值）成立，REML 共用同一邊界，實證確認其停在下限。

**A3**：`run_synthetic_resolution.py` 的 480 格網格（m_g∈{10,25,50,100,250} ×
τ²/σ²∈{0.02…1.0} × w∈{1,0.99,0.95,0.90} × equal/unequal n_i × homo/hetero σ²，
每格 4000 次模擬）。**median |err| = 0.0006、90th = 0.040、max = 0.207**。
誤差集中於「小群 + 計數不等 + 變異異質」，且該處近似**低估** collapse 機率——
已寫入正文與 Limitations，條件明示而非假設。

**A4**：`identifiability limit` → **`variance-component resolution condition`**；
刪除 `no amount of estimation effort removes`；Corollary 後新增
「**What this says and does not say**」段，明寫這是 estimator- 與 window-specific 的
estimability 陳述，**不是** information-theoretic impossibility，且不說明門檻以下哪個 partition 最好。

**A5**：兩個問題都成立。改為以 **item 為獨立單位**（每 item 先聚合成一個 validation loss），
n=3649（OR）而非 ~8×10⁴ 個 observation-quantile 對；有界性明寫為對 per-item loss 的
**clipping 假設**（per-series scaling 的分母）。以 M=24、η=0.05、n=3649 計，penalty = 0.061·B，
而 partition 間 SPL 實測差僅 ~0.004 —— **界不足以區分 partition**，這與實證結論一致，
稿中據此把它當佐證而非賣點。oracle inequality 在貢獻段**降級為 supporting result**。

---

## B. Synthetic experiment — ⚠️ 本節結論**已於 2026-07-30 撤回**，見下方 B-REVISED

| 項 | 內容 | 狀態 |
|---|---|---|
| B1 | controlled pooling–forgetting resolution 實驗 | **DONE** |
| B2 | 三面板核心圖 | **DONE** |
| B3 | 區分 plug-in failure / partition recovery / information shortage | **DONE** |

`src/experiments/run_synthetic_resolution.py`（決定性，seed 參數化）。
網格：separation∈{0.25,0.5,1,2} × items/group∈{40,120} × w∈{1,0.99,0.95,0.90} × **5 重複**，
六個 arm（no-pooling / global-stationary / global-discounted / oracle-partition /
learned-partition / learned+regularized）。輸出 `outputs/synthetic_resolution/`，
圖 `paper_v2/figs/fig_resolution.pdf`，正文新節 §`sec:synthetic`。

**三方判別結論（每格 n=10 配對重複）**：
- (a) **pooling 本身很值錢**：single pool 全格勝過 no-pooling，優勢隨異質性增加（0.010→8.13）。
- (b) **mixture 找得到結構**：ARI 由 0.26（sep=0.25，選 K≈2）升到 **0.95–0.96（sep=2.0，選 K=4＝真值）**，
  且幾乎不受折扣影響 → **不是 recovery 失敗**。
- (c) ~~連 oracle partition 都不划算 … 是 information shortage~~
  **← 此項已撤回**：受 separation/難度混淆與只看 MAE 兩個缺陷影響，見 B-REVISED。
  (a) 與 (b) 不受影響，仍成立。

**過程中修掉的兩個實作缺陷**（首版跑出的結果無效，已作廢）：合成資料的群只在 size mean 上分離，
BIC 因此選 K=1、learned 與 global 數值完全相同；local arm 讓每個 item 自成一群導致
變異成分未定義、預測值爆到 1e98。已改為群在 size 與 occurrence **兩個 block 上分離**，
並以 per-item 經驗 hurdle 取代 local arm。

---

## B-REVISED（2026-07-30）：B3 結論**已撤回並更正**

外部審閱對 B3 提出六個可能的 artifact。逐項查證後，**其中兩個成立，B3 原結論作廢**。

**成立的缺陷**
- **#5 separation 與難度混淆**：生成器把群心放在 0, s, 2s, 3s，separation 一大，
  panel 的 grand mean 與需求尺度同步變大——`global-discounted` 的平均 MAE 由
  **0.726 → 1.236 → 4.236 → 67.279**。格點根本不可比，「差距隨 separation 擴大」大半是尺度假象。
- **#4 只看 MAE**：同一批 run 改看 pinball，結論反轉。oracle partition 在
  **w=0.90 勝 2.06–3.22%、10/10 配對重複全勝**；w=0.99/1.00 才略輸（−0.1% 至 −1.2%）。
- **#2 oracle 不完整**：舊 arm 只給真實標籤仍重估群超參數。已新增
  `fit_tsb_hb(fixed_group_hypers=...)` 與完整階梯（global / labels+estimated /
  labels+true / true predictive）。

**不成立的**：#1 符號方向正確（已逐格核對原始 MAE）；#6 各 arm 的 train/test 與 refit protocol 相同。

**新的、已驗證的結論（sanity check 4，`outputs/synthetic_sanity/extreme.csv`）**
舊的「極端可辨識案例」其實**資料太充足**：T=400 時每個 item 有 154 個正觀測，λ_i≈1，
任何先驗都不起作用——那是檢查本身失敗，不是模型。改為**固定結構、只掃序列長度**後：

| T | median n⁺ | single pool MAE | oracle labels | +true hypers | oracle 增益 |
|---|---|---|---|---|---|
| 400 | 154.5 | 5.2006 | 5.1987 | 5.1999 | 0.04% |
| 60 | 23.5 | 5.1682 | 5.1525 | 5.1567 | 0.30% |
| 25 | 10.0 | 5.3103 | 5.2302 | 5.2355 | **1.51%** |
| 12 | 4.5 | 5.5383 | 5.3488 | 5.3439 | **3.42%** |

pinball 同向（0.01% → 1.53%）。**oracle partition 確實會贏，而且贏多少由每個 item 自己的
樣本數決定，不是由群間距離決定**——正是 credibility 權重 λ_i=n_i/(n_i+κ_g) 所說的。
給真實超參數幾乎不改變結果 → **超參數估計不是瓶頸**。true-predictive 全場最佳（正確性檢查通過）。

**因此正文改為**：learned partition 在五個面板上不動總體指標，**不是**因為結構不存在或找不到，
而是這些面板的中位 item 自身歷史已足夠（single pool 下 median λ_i 0.75–0.90），先驗只是次要項。
「information shortage」的措辭已從論文移除。

**仍存在的已知缺陷（未修，數字未使用）**：`run_synthetic_sanity.py` 的
`true_predictive()` 用 item 的初始 occurrence p_i，未跟隨生成器的 drift，
因此**啟用 drift 的 grid（checks 1–3）無效**——證據是該 grid 中 true-predictive 在
MAE/pinball/Brier 上最差、卻在 log_size_mse 上最好（1.6304，確實最低），正是 occurrence
oracle 錯、size oracle 對的指紋。`extreme` 案例 drift=0，不受影響。
**修法**：true_predictive 需取測試期各時點的 p_t。修好前 grid 數字不得入文。

---

## C. learned pooling 的實證補強

| 項 | 內容 | 狀態 |
|---|---|---|
| C1 | 五資料集完整 pooling grid | **PARTIAL** |
| C2 | selector 的 validation evidence + CI | **PARTIAL** |
| C3 | cold-start slices 四方比較 | **PARTIAL** |
| C4 | mixture component 可解釋性與穩定性 | **PARTIAL** |

- C1：OR 六格（含 SPL/q90/AIW）在 `tab:ablation`；Auto/Carparts/RAF 的 **point** 六格在
  `ablate_*`。**缺**三個月頻面板的 **prob** 六格與 M5 全格（純機器時間，指令見文末）。
- C2：`remix_selection_diagnostics.csv` 已含每個候選的 validation 分數；
  **缺** bootstrap CI 與「mixture 領先是否超過 validation noise」的正式檢定。
  但 A5 的界已間接給出答案（0.061·B ≫ 0.004 的實測差），正文已據此改口徑。
- C3：現有 bin CS_0_1/2_3/4_5/6_10/11_plus，已有 global vs learned 與 learned+regularization
  （四面板）。**缺** taxonomy 欄、SPL/q90 欄、CI，以及依 effective sample size / occurrence rate 分 bin。
- C4：component 特徵表在 `tab:components`。**缺** responsibility entropy、跨 validation-ratio 的
  component matching、BIC 曲線圖。

---

## D. Baseline fairness

| 項 | 內容 | 狀態 |
|---|---|---|
| D1 | tuned TSB（同一 chronological split） | **PARTIAL（程式完成，實跑進行中）** |
| D2 | 不可宣稱勝過未跑過的 TweedieGP | **DONE** |
| D3 | hierarchical dynamic baseline | **BLOCKED → Limitations 已明說** |
| D4 | conformal 不可以偏概全 | **DONE** |

- **D1**：`src/models/tsb_tuned.py` 完成——`tune_tsb_on_split()` 用**與 REMIX 相同的**
  `_split_init_head_tail`（同一 80/20 時序切分）、5×5=25 個 (α_d, α_p) 候選（與 REMIX 的 24 個同量級）、
  以 scaled MAE 為準則（避免各自在自己主場評分）。OR 實跑進行中，數字尚未入文。
  **入文前必須完成**，並同時報 fixed TSB / tuned-by-MAE / tuned-by-scaled 三列。
- **D2**：全文掃描確認**原本就沒有**「outperforms TweedieGP」之類宣稱；但 baseline 名稱
  `Tweedie` 易被誤讀，已全面改稱 **Tweedie-GLM**，正文加註「非 \citet{damato2025} 的 GP 方法」，
  Limitations 第 (7) 點明寫未跑、不作相對宣稱。
- **D4**：`collapse at upper quantiles` → `degrade sharply under the present split-conformal
  construction`，並明寫 adaptive / rolling / weighted conformal 未評估、結論僅限本文的 wrapper。

---

## E. Evaluation 修正

| 項 | 內容 | 狀態 |
|---|---|---|
| E1 | degenerate baselines 措辭 | **DONE** |
| E2 | RAF saturation → 摘要口徑 + 遠尾指標 | **PARTIAL** |
| E3 | criterion alignment 升級為正式實驗 | **PARTIAL** |
| E4 | walk-forward configuration 對齊 | **PARTIAL** |
| E5 | significance tests 雙向敘述 | **DONE** |
| E6 | cross-dataset 檢定的方法假設 | **DONE** |

- E1：改為「MAE on highly intermittent panels rewards conservative zero-leaning forecasts and
  should be interpreted jointly with scaled-error and distributional metrics」，不再用來對比 TSB。
- E2：摘要與 §4.2 已改為 **best among non-degenerate methods**，RAF 飽和 caveat 入文。
  **缺** q95/q97.5/q99、CRPS、newsvendor cost 的實跑。
- E3：已建立正式小節 §`sec:criterion`（MAE 準則選出 (global, 0.90)，OOS MAE 5.5083 優於
  表中所有配置與 baseline，含 TSB 的 5.5736），並雙向陳述（支持設計，同時承認主配置非其族中 MAE 最優）。
  **缺**完整 regret matrix（5 準則 × 5 指標 × 5 面板），需新增 `--selection-criterion` 參數。
- E4：已明確區分 initialization discount 與 online block update，聲明**未做 double-discounting**，
  並把 walk-forward 定位為「fixed-origin 配置的序列化操作」。**缺** walk-forward-selected
  configuration 與 re-selection cadence 的實跑。
- E5：§4.3 對 TSB 已是雙向陳述（median diff +0.29 TSB 優、mean diff −0.110 我方優、
  Wilcoxon 顯著 / t 不顯著），不以一句 beats TSB 概括。**缺** win rate 與 bootstrap CI 欄。
- E6：Friedman/Nemenyi 的 block 應為 dataset 而我們用 series，且五個面板序列數 2,509–5,000
  造成權重偏斜；CD 圖與 DM 檢定**已降級為 descriptive**，圖說與正文明寫假設不滿足、
  DM 用於 per-series aggregate loss 屬非標準用法。

---

## F–N. 寫作與定位 — 全部完成

| 項 | 內容 | 狀態 |
|---|---|---|
| F1–F5 | Introduction 重寫 | **DONE** |
| G1–G3 | Contributions 三段重寫 | **DONE** |
| H | Abstract 重寫（實測 **220 words**） | **DONE** |
| I1–I3 | Related Work 修正 | **DONE** |
| J | Theory 加「說了什麼／沒說什麼／哪個實驗測它」 | **DONE** |
| K1–K4 | Results 措辭修正 | **DONE** |
| L | Limitations 精簡為四類 | **DONE** |
| M1–M3 | 命名與術語統一 | **DONE** |
| N | 全文主張統一 | **DONE** |

- F1：`dominates`→`is pervasive in`、`knows that`→`is designed to adapt`、
  `specifies no probabilistic model`→`does not natively provide a full predictive distribution`；
  不再宣稱所有 hierarchical model 皆 stationary。
- F2：刪除 `Each tradition hard-codes what the other learns`，改為
  「commonly pre-specify at least one of two structural choices… we represent these choices within a
  common model family and select candidate pairs by chronological validation」。
- F3：圖說改為「pooling structure 與其 EB 收縮**共同**決定水平位置」，明說不是直接 grid-search φ。
- F4：刪除 `generic, composable operators` 與 `at zero additional cost`，改為
  `two compatible modifications` / `without changing the asymptotic cost of a single fit` /
  `linear in the number of series for a fixed candidate set`。
- F5/N：主線改為指定的非對稱敘事，並在 Introduction 收尾以
  「Forgetting preserves the need for pooling, but not necessarily the information needed to
  estimate a fine pooling structure.」定調。
- M3：TSB exact claim 全面附初始化條件（`exact under a self-consistent initialization,
  with $O(w^T)$ discrepancy otherwise`）。

---

## 尚待機器時間的指令（依優先序）

```bash
# D1 收尾（最高優先，投稿前必辦）：把 TSB-tuned 列入主表
#   已完成 src/models/tsb_tuned.py；需接進 run_point 的 baseline 清單並五面板重跑

# C1 缺口：三個月頻面板的 prob 六格
for ds in auto carparts raf; do for g in global taxonomy mixture; do for w in 1.0 auto; do
  py -m experiments.run_prob --dataset $ds --baseline-mode hb_only --hb-grouping $g \
     --hb-fit-discount $w --hb-calibration-mode none --hb-bootstrap-draws 20 \
     --out ../outputs/aistats2027/ablate_${ds}_prob_${g}_${w}
done; done; done

# E2 缺口：遠尾分位數 —— 在上列指令加 --quantiles 0.5,0.75,0.9,0.95,0.975,0.99

# E3 缺口：cross-objective regret matrix
#   需在 select_pooling_and_discount 增加 --selection-criterion {spl,mae,rmsse,q90,newsvendor}

# E4 缺口：walk-forward re-selection —— 需在 run_point 加 --reselect-every N

# C2/C3/C4 缺口：bootstrap CI、taxonomy 欄與 SPL 欄、responsibility entropy 與 BIC 曲線
```
