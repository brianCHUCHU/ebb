# DESIGN — Pre-fit pooling-room diagnostic（預註冊，2026-09-15）

## 問題
Local vs pooled forecasting 通常只能事後比較。我們要驗證：**只 fit 最粗的 global EB、不 fit 任何
partition**，能否從 fitting window 的量事先判斷「細分 pooling 是否還有值得追求的 room」。
主張是 screening（有沒有 material room），不是預測回報幾個百分點。

## 特徵（全部來自 global-pool fit，fitting window）
size block：
- `lev_size` = median_i (1 − λ⁺_i)，λ⁺_i = n⁺_i/(n⁺_i + σ̂²/τ̂²)（global 的 κ）
- `het_size` = max(0, S² − s̄)：S² = 品項 mean_log 的樣本變異（n⁺≥1），s̄ = mean_i σ̂²/n⁺_i
- `z_size` = (S² − s̄)/(√(2/(m−1)) S²)（Prop 2 的 resolution 統計量，m = 品項數）
occurrence block：
- `lev_occ` = median_i (1 − λ⁽ᵒ⁾_i)，λ⁽ᵒ⁾_i = n_i/(n_i + α̂ + β̂)
- `het_occ` = max(0, S_p² − s̄_p)：p̂_i = s_i/n_i，s̄_p = mean_i p̂_i(1−p̂_i)/n_i
- `z_occ` 同構
其他：m（品項數）、median n⁺、w、T（後兩者為協定已知量，允許使用）。

## Room score（事前寫定）
- **R1（閉式）** = lev_size² × het_size + lev_occ² × het_occ（兩 block 相加，各自「無槓桿則無用、無異質則無用」）。
- **R2（配適）**：對 log(1+特徵) 的 logistic 迴歸（分類 Δ>ε）與線性迴歸（Δ），只用上列特徵；在 held-out
  訓練摺以外的格上配適。R2 的目的在回答「可學到多少」，主張以 R1 為主、R2 為上限。

## Target
Δ_oracle = 100 × [SPL(global) − SPL(true labels, estimated hypers)] / SPL(global)，pinball q∈{.5,.75,.9}
在測試段（與 run_synthetic_sanity 同評分）。負值照實保留。

## Synthetic 網格（`r1_room_synthetic.py`，seed 20260915）
separation ∈ {0, .25, .5, 1, 2} × w ∈ {1, .95, .90} × T ∈ {30, 60, 120} × occurrence mean ∈ {.1, .25, .5}
× items/group ∈ {20, 60} × 組大小 ∈ {均衡, 失衡 70/10/10/10}；4 組；τ²=0.15、σ²=1、drift 0.5、
occurrence spread = 0.8·min(sep,1)（第二版曲面慣例）；5 replicate/格 → 540 格、2,700 panel。

## Held-out 評估（`r2_room_eval.py`）
三種整塊留出：leave-one-separation-out、leave-one-w-out、leave-one-T-out（各層輪流），
另加隨機 5-fold 作對照。指標：Spearman(score, Δ)、MAE（R2 迴歸）、**主要：AUC 與
precision/recall 於 Δ>0.5% 與 Δ>1%**（R1 用 rank 作 AUC；R2 用 held-out 機率）。

## Semi-synthetic transfer（`r3_room_semisynthetic.py`）
真實序列（Online Retail、Carparts、M5 5,000 子樣本），隨機分 4 組（3 個 seed），對全序列正值乘
exp(offset_g)，offset 對稱 ±0.5δ、±1.5δ，δ ∈ {0, .25, .5, 1, 2}；occurrence 不注入（真實 zeros 模式不動）。
w 用審定值；評估區間用真實 fixed-origin 切分。R1 直接算；R2 用 synthetic 全部格配適後直接套用（零 transfer 調參）。

## 判讀（事前寫定）
- held-out AUC(Δ>1%) ≥ 0.8 且 semi-synthetic AUC ≥ 0.75：Intro 升級為「診斷能在擬合任何 partition 前預測
  refinement 的 oracle 增益」。
- 0.65–0.8：寫成「能篩掉無 room 的 regime」。
- 更低：維持現狀，診斷只當上界，列 limitation。
兩階段順序：synthetic 先跑並評估完成後，才跑 semi-synthetic；不回頭改 score 定義。

## 執行後的修正紀錄（2026-09-15，全部與預註冊版並列報告）
1. **R1 的 s̄ 在折扣下失效**：未加權的 mean_i σ̂²/n⁺_i 被 n⁺→0 的品項主導（OR 真實資料 s̄=12.5 > S²=5.9，het=0）。
   修正版 R1' = med(1−λ⁺)² × τ̂²_global（模型自身的加權動差估計）。R1 照原定義報（AUC≈0.5）。
2. **global pool 塌陷**：網格 39% 的 panel z_size ≤ 0（單一 pool 的 τ̂² 塌到 0），且 Δ>1% 的案例中 52% 屬此類——
   這類「room」是逃離塌陷的 global fit，不是結構的回報。全網格與「resolved」子集都報。
3. R2 另跑「sound」特徵集（去掉 S²/s̄/z 類），AUC 較低（0.76–0.78），預註冊特徵集為主報。
4. semi-synthetic 在審定 w 下無正例（Δ ≤ 0.39%）：注入的分離度被 global τ̂² 吸收、λ→1。加跑 w=0.9 regime
   仍無正例（Δ ≤ 0.27%）。判讀：真實序列在有足夠正值時，細分 pooling 沒有 room；診斷對其全部指派 P(Δ>1%) ≤ 0.17（正確拒絕），
   但無法在真實序列上檢驗偵測力。
判定（依事前規則）：held-out AUC 0.82–0.84（留出分離度/T/隨機）、0.68（留出 w）→ 「能篩掉無 room 的 regime」層級，
不升級為「預測 oracle 增益」。
