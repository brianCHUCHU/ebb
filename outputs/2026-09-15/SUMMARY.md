# SUMMARY — 2026-09-15（pre-fit pooling-room screen；判定「能篩掉無 room 的 regime」層級）

設計 `docs/DESIGN_room_diagnostic.md`（含執行後修正紀錄）。腳本：`room_features.py`、`r1_room_synthetic.py`、
`r2_room_eval.py`、`r3_room_semisynthetic.py`、`r4_room_transfer.py`、`r5_room_rows.py`、`w17_room_tex.py`。

## Synthetic 網格（2,700 panel，17 分）
sep{0,.25,.5,1,2}×w{1,.95,.9}×T{30,60,120}×occ{.1,.25,.5}×items{20,60}×{均衡,失衡}×5 rep。
target Δ_oracle 分布：>1% 佔 22%、>0.5% 佔 29%；w=0.9 平均 +0.63%、w=1 −0.68%。
**39% 的 panel 單一 pool τ̂² 塌陷（z≤0），Δ>1% 案例中 52% 屬此類**（逃離塌陷，非結構回報）。

## Held-out AUC（Δ>1%）
| 留出 | R1（預註冊） | R1'（τ̂² 修正） | R2（logistic） | R1' resolved | R2 resolved |
|---|---|---|---|---|---|
| 分離度層 | 0.45 | 0.60 | **0.84** | 0.72 | 0.85 |
| w 層 | 0.64 | 0.67 | 0.68 | 0.66 | 0.68 |
| T 層 | 0.51 | 0.71 | 0.82 | 0.79 | 0.83 |
| 隨機 5 折 | 0.51 | 0.71 | 0.83 | 0.78 | 0.84 |
R2 精確率/召回率（best-F1）約 0.54 / 0.65–0.77，基準率 22%。「sound」特徵集（去掉 S²/s̄/z）R2 降到 0.76–0.78。

## 修正（與預註冊版並列報告）
R1 的未加權 s̄ 在折扣下被 n⁺→0 的品項主導（OR：s̄ 12.5 > S² 5.9 → het=0）→ R1' 改用模型加權 τ̂²。

## Semi-synthetic transfer（OR、Carparts、M5；δ∈{0,.25,.5,1,2}，3 assignment；審定 w 與 w=0.9）
**無正例**：oracle 增益最大 0.39%（M5, δ=2）；注入的分離度被 global τ̂² 吸收（Carparts 0.04→5.1）、λ⁺→1。
R2 對所有注入面板 P(Δ>1%) ≤ 0.17（正確拒絕）；偵測力無法在真實序列上檢驗。
機制含義：真實序列在有足夠正值時細分 pooling 沒有 room；room 只在低 n⁺ regime（短歷史／強遺忘）。

## 判定（事前規則）
held-out 0.82–0.84（留出分離度/T/隨機）、0.68（留出 w）→ **「能篩掉無 room 的 regime」**；不升級為「預測 oracle 增益」。

## 稿件（v6）
`sections/room_main.tex`（§5.2 新段「A pre-fit screen」＋fig:room）、`sections/room_app.tex`（app:room：協定、tab:room、
fig:room-transfer）；abstract/Intro/貢獻(1)/Limitations(2) 各加一句。builder 已加 \input 保留。38 頁、0 錯誤；
正文到第 10–11 頁，需裁約 2 頁（fig:room 可退附錄省 0.5 頁）。abstract/Intro 的 w17 補句只在 v6/main.tex，builder 重跑會丟，
需再套 w17。

## 落版壓縮（作者裁決）
正文只留一段「A pre-fit screen」（無圖）；fig:room 與 fig:room-transfer、tab:room、協定與修正紀錄全在附錄 app:room；
abstract 補句刪除；Intro 補句縮成半句「a screen built from the single-pool fit identifies, in simulation, the regimes
where refinement cannot pay materially」；貢獻 (1) 子句與 Limitations (2) 補句保留。`w17b_compress.py` 已改寫 w17，重跑一致。
v6：38 頁、0 錯誤、References 回到第 10 頁（與加入前相同）。
