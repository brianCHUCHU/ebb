# 設計規格：品項各自的遺忘率 w_i（EB 收縮）

> 2026-09-05 草擬。目的：對準 TweedieGP 唯一的實質優勢（per-series 時變適應），
> 在不失閉式、不動 pooling/理論骨架的前提下，讓遺忘率隨品項變化。
> 執行時機：等 M5 wf 與五面板 wf roster 跑完後，先做閘門評估再決定是否採用。

## 1. 現況與問題
- 現在 w 是整個面板一個純量，由 head/tail（80/20）驗證聯合選出。
- TweedieGP 每條序列各自學 lengthscale ＋ 偏近期的 inducing points，等於 per-series 的 w。
- 診斷（quantile_coverage_diag、predictive_law_variants）顯示差距不在分布形狀、不在推論方法，
  在後驗的時變適應。

## 2. 方法（"pooling of forgetting"）
記 b = 1 − w（遺忘強度），品項 i 的驗證損失曲線 R_i(w)（在 8 點網格上）。

**Step A — 內層切分算每品項的損失曲線**
- 沿用既有 `_split_init_head_tail`：外層 head_o / tail_o（80/20）。
- 在 head_o 內再切 head_i / tail_i（80/20，min_head 8）。
- 對每個 w ∈ W：以 head_i 擬合（既有 `_score_discount_grid` 的路徑），對 tail_i 的每列算
  pinball（q ∈ {.5,.75,.9}，除以 head_i 的 a_i），**按品項聚合** → R_i(w)、n_i^V（tail_i 列數）。
  （現有程式已算到逐列 pinball，只差不要在品項層平均掉。）

**Step B — 收縮：品項曲線與面板曲線的 precision-weighted 組合**
- 面板曲線 R̄(w) = 品項平均。
- 品項 i 的收縮目標函數：J_i(w) = n_i^V · R_i(w) + κ · R̄(w)
- w_i = argmin_w J_i(w)。κ 是唯一新增的純量（pseudo-count）：κ→∞ 退回現行單一 w，
  κ=0 是每品項各自 argmin（短序列會過擬合）。
- κ 在小網格 {∞, 300, 100, 30, 10, 3} 上以**外層 tail_o** 選（與結構選擇同一層、同一準則），
  避免用定義 R_i 的資料選 κ。
- 這正是論文自己的 credibility 形式（n/(n+κ)）用在遺忘軸上，敘事上與 Prop 2 同構。

**Step C — 擬合與預測**
- `fit_tsb_hb(..., fit_discount=<per-item Series>)`：sufficient statistics 對每個品項用自己的
  w_i^{T_i−t} 加權；群層超參數由各品項的折扣統計量估計（「同一運算子」變成
  「各品項自己的運算子，貢獻到群統計量」）。純量 w 仍為特例（全部相同）。
- 選擇流程：結構 r 與 κ 在 tail_o 聯合選（候選數 3 × 6 = 18；每個候選需要 8 個 w 的
  內層擬合，已是現行成本量級）。
- 線上更新（wf）：各品項用自己的 w_i 累積，機制不變。

## 3. 理論影響
- Prop 1（TSB 特例）：逐品項成立（w_i 對應各品項的 α_p）。
- Prop 2 / Cor（雙邊 credibility、resolution）：n_i(w) → n_i(w_i)，證明逐字不變；
  群層 s̄_g 的下界改為 min_i (1−w_i)σ_g²。
- Prop 4（有限候選 oracle）：候選數 24 → 18（結構×κ），但每候選內含品項層選擇——
  嚴格說品項層選擇是 N 個小決策，需說明其 oracle 界靠 κ 收縮控制（誠實寫進 limitation）。

## 4. 閘門評估（fixed-origin，五面板，baseline 不重跑）
| 量 | 判準 |
|---|---|
| mean SPL vs 現行 EBB | OR 或 M5 改善 ≥ 1%，且 Auto/RAF 不惡化超過 0.3% |
| 對 TweedieGP | 至少收復 OR（0.8845）或把 M5 差距壓到 <1.5% |
| κ=0（無收縮）與 κ=∞ 的對照 | 曲線需呈內點最優（否則等於沒學到東西） |
| 額外成本 | 選模時間 ≤ 現行 3× |
未過閘門 → 寫成 negative result 段（"per-item forgetting does not close the gap"），不採用。

## 5. 採用後的連動（估 2 天機器＋2 天稿）
EBB 列（fixed + wf 五面板）、selection surfaces、λ 表（λ 在各自 w_i 下計算，
語義改為「at the item's selected discount」）、fig10a/b、SPL 顯著性、效率表、M5 三 seed、
§3.4 選擇流程文字、Prop 2 附註、Limitations。
