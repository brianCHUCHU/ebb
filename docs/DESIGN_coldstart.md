# DESIGN — 短歷史（cold-start）縮放實驗（預註冊，2026-09-07）

## 目的
Prop 2 / Cor 1 的機制預測：品項自身資料越少（credibility λ 越低），共享先驗的
槓桿越大，pooling 的價值越高。本實驗以**截短初始窗**製造 λ 的連續變化，
在固定的外部評估區間上比較 pooled（EBB）與 per-series 方法（TweedieGP、
CP-ADIDA、CP-Croston）。這是證據實驗，不是採用閘門；結果不論方向照實入文。

## 假設（事前寫定）
H1. 隨初始窗長度 L 縮短，median λ⁽⁺⁾ 與 λ⁽ᵒ⁾ 單調下降。
H2. EBB 對 per-series 方法的相對優勢（(SPL_cmp − SPL_EBB)/SPL_cmp）隨 L 縮短而
    單調增加；在最短的 L，EBB 在每個面板領先 TweedieGP。
H3. 在全長 L，結果與已發布數字一致（sanity）。
若 H2 不成立（例如 TweedieGP 在短窗仍領先），寫成 negative result。

## 協定
- 資料與評估區間：與 fixed-origin 主實驗完全相同（OR init 1/3、M5 init 2/3 於
  seed-42 的 5,000 條、monthly last-h）。評估區間不變。
- 截短：每條序列只保留初始窗**最後** min(L, n_i) 個觀測。L 網格：
  - Online Retail（init 中位 119 天）：14, 28, 56, full
  - M5（init ≈ 1,294 天）：28, 56, 112, 224, 448, full
  - Auto（18 月）：6, 12, full
  - Carparts（45 月）：6, 12, 24, full
  - RAF（72 月）：6, 12, 24, 48, full
- EBB：審定的 (structure, w) 固定不重選（OR global .95、M5 mixture .95、
  Auto global .99、Carparts global .90、RAF mixture .997）；mixture 標籤在截短窗
  上重新學習（partition builder 只看得到截短窗）；B=20、seed 42，與已發布列同設定。
- TweedieGP：released defaults，同 h2 runner（T≤200 全點 inducing、否則 200/log）。
- CP-ADIDA / CP-Croston：`fit_predict_conformal_baselines`，cal_ratio 0.2；短窗下
  失敗照記錄。
- Zero：診斷列。
- 尺度：scaled pinball 的分母固定用**全長**初始窗的 naive scale，使不同 L 可比。
- λ：`scripts/analysis/leverage_dual_lambda.dual_lambda(init_trunc, w)`（不改）。

## 輸出
`outputs/<date>/coldstart_<panel>.csv`（panel, L, model, spl_mean, spl_q90,
n_series, median_lambda_occ, median_lambda_size）、`coldstart_all.csv`、
`c1_run_meta_<panel>.json`；圖：x = L（log），y = EBB 對 TweedieGP 的相對優勢，
副軸標 median λ⁽⁺⁾。

## 不做的事
不重選 (structure, w)；不改模型；不對 M5 以外做 seed 變異；不用 outputs/aistats2027/。
