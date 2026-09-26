# DESIGN — occurrence 與 size 分開的遺忘率（two-rate forgetting，預註冊，2026-09-07）

## 動機
M5 的病灶已診斷在 occurrence block：EBB 的 q90 在 20.6% 的序列歸零（TweedieGP 13.4%），
而預測律形狀變體與 per-item w_i 都無法收復。TSB 本身就有兩個平滑常數（occurrence 與 size
各一）；EBB 目前在初始窗把同一個 w 套在 occurrence 計數（n, s）與 size 統計量
（n⁺, Σlog, Σlog²）上。本實驗讓 occurrence 有自己的折扣 w⁽ᵒ⁾，size 維持審定的 w。

## 模型層改動（最小）
`_compute_series_stats(train_df, group_labels, fit_discount, occurrence_fit_discount=None)`：
- `occurrence_fit_discount is None` → 與現行完全相同（純量路徑必須位元不變；以
  `outputs/2026-09-05/scalar_path_reference.txt` 的 sha 驗證）。
- 否則 n_obs、s_obs 用 w⁽ᵒ⁾^lag 加權；n_pos、sum_log、sum_sq_log 用 w^lag 加權。
- `fit_tsb_hb` 與 `initialize_online_tsb_hb` 透傳該參數；線上更新路徑不改
  （wf 實驗中 occurrence 折扣僅作用在初始窗，與現行 size 折扣的處理一致）。
- 理論：Prop 1（TSB 特例）自然對應雙 α；Prop 2 的 n_i(w) 在 occurrence block 改為
  n_i(w⁽ᵒ⁾)，證明逐字不變。

## 選擇（leakage-safe）
審定的 (structure, w) 固定；w⁽ᵒ⁾ 在同一個 80/20 初始窗 head/tail 切分上、以同一
`_score_discount_grid` 準則（q ∈ {.5,.75,.9} 的 scaled pinball）從網格
{tied（= w）, 0.999, 0.997, 0.995, 0.99, 0.98, 0.95, 0.90, 0.80, 0.70} 選出。
tied 在網格內，所以選擇器可以拒絕改動。

## 閘門（fixed-origin 外部評估，B=20，五面板）
| 量 | 判準 |
|---|---|
| M5 mean SPL | 改善 ≥ 1%（目標病灶） |
| 其他四面板 | 各不惡化超過 0.3% |
| 選擇器 | 至少 M5 選到 w⁽ᵒ⁾ ≠ tied（否則等於沒學到） |
| M5 q90 歸零率 | 下降（診斷量，非硬判準） |
未過 → negative result 段，不採用。過 → 採用為 EBB 的一部分並重跑全部連動
（fixed + wf 五面板、selection surfaces、λ 表、顯著性、效率表、三 seed、§3 文字）。

## 輸出
`outputs/<date>/occdisc_selection.csv`（面板 × 網格 tail 分數）、`occdisc_eval.csv`
（外部 mean/q90、q90 歸零率）、`o1_run_meta.json`。
