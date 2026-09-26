# DESIGN — 分離度軸：把 Figure 10 的 envelope 從曲線變成曲面（預註冊，2026-09-13）

## 問題
§5.5 的模擬 envelope 只沿 leverage（w, T）掃；grid.csv 其實含 separation ∈ {0.5, 1, 2}
但被當成 9 個 replicate 混在一起，長度掃描則固定 3.0。真實面板只用 median λ⁽⁺⁾ 一個
座標定位，所以「Online Retail 位在真 partition 值 >2% 的區域」無法與「真實組間距離
太近、room 本來就小」區分。

## 模擬曲面（`s1_separation_surface.py`）
生成器不改（`run_synthetic_sanity.simulate`，4 組 × 60 品項、τ²=0.15、σ²=1、
occurrence_mean 0.25、occurrence_spread 0.8、grand_mu 1、drift 0.5）。
網格（事前寫定）：separation ∈ {0, 0.25, 0.5, 1, 2, 3}；w ∈ {1, 0.95, 0.90}；T ∈ {30, 120}；
9 個 paired replicate；seed 20260913。
每格記錄：
- 真 partition 的 paired pinball 增益（oracle-labels+estimated-hypers 對 global），mean 與 SE；
- **learned 增益**：新增 arm `mixture-learned`（`mixture_group_labels(train, k=0, w)`，
  與正文相同的目標函數）對 global 的 paired pinball 增益——模擬裡目標函數在哪個分離度
  開始找得到結構；
- median λ⁽⁺⁾（global arm）；
- **分離度指標 R²_size**：訓練窗品項 log-size 均值（同一 w 的加權統計量）對「真標籤」
  的 ANOVA R² = 1 − SS_within/SS_total（含抽樣噪音，與真實面板同定義），以及對 learned
  標籤的 R²。

## 真實面板（`s2_real_separation.py`）
五面板、審定的 (structure, w)、與 λ 表相同的初始窗。品項 mean_log 來自
`_compute_series_stats(init, fit_discount=w)`。兩套標籤：
(i) learned mixture（`mixture_group_labels(init, k=0, w)`，即正文使用的那一個）；
(ii) taxonomy（ADI/CV² 四類，與目標函數無關）。
各算 R²_size、組數、各組 mass 與中心（附錄用）。**不用 τ̂²** 作分母（7 月已證明
learned 下 τ̂² 塌到下限）。

## 定位與判讀（事前寫定）
以 (median λ⁽⁺⁾, R²_size) 兩座標把五面板放到曲面上；在該座標附近的模擬格讀出
「真 partition 的增益」＝該面板的 room。判讀規則：
- room ≥ 2% 且 learned 增益 ≤ 0.6%：維持「learnability binds」的敘述，並以
  room − realized 量化每面板的 learnability gap；
- room < 1%：改寫為「診斷（含分離度）事先就說沒有 room；負結果是預測的結果」，
  §5.6 的「Online Retail is the decisive case」降級。
- 介於其間：如實寫成部分。
模擬先跑、真實面板再估、最後才定文字；順序寫進附錄。

## 輸出
`outputs/<date>/sep_surface.csv`（每 rep）、`sep_surface_cells.csv`（每格彙總）、
`real_separation.csv`、`fig14_separation_surface.{pdf,png}`、`tab_separation.tex`。

## 附錄（2026-09-13 加跑，事前寫定）
第二版曲面（`s1b_separation_surface_v2.py`）：
(3) occurrence 分離度跟著 size 分離度縮放：occurrence_spread = 0.8·min(sep, 1)，
    使 sep=0 在兩個 block 都無結構；sep ≥ 1 的格子與第一版完全相同（同 seed）。
(2) learned mixture 的兩個已發布修法作為額外 arm：label sample-splitting（parity）、
    credibility hyper-shrink，各自與合併。判讀：若某修法在高分離度收回真 partition
    增益的可觀比例，寫成「失敗機制已知、修法在模擬中收回 X%」；否則維持 negative。
