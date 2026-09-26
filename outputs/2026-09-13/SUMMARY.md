# SUMMARY — 2026-09-13（分離度軸：envelope 曲線 → 曲面；核心 claim 修正）

## 起因
審稿式回饋：§5.5 envelope 只沿 leverage 掃、分離度固定；OR 只用 λ⁽⁺⁾ 定位，
「OR 位在真 partition >2% 的區域」無法與「真實組本來就靠得近」區分。
實查更糟：舊 fig10 的灰線是把 grid.csv 的 separation 0.5/1/2 混成 9 個 replicate 平均，
長度掃描用 3.0——分離度根本沒被控制。

## 做了什麼（`docs/DESIGN_separation.md` 預註冊；模擬先跑、真實面板後估、最後定文字）
- `s1_separation_surface.py`：同一生成器，sep ∈ {0,.25,.5,1,2,3} × w ∈ {1,.95,.9} × T ∈ {30,120}，
  9 replicate，seed 20260913；每格記真 partition 增益、**learned mixture 增益**（新 arm，正文同一目標函數）、
  global-pool median λ⁽⁺⁾、真標籤 R²_size。36 格、31 分。
- `s2_real_separation.py`：五面板在審定 w 的 R²_size（品項 log-size 均值被分組解釋的比例），
  learned mixture 與 taxonomy 兩版。不用 τ̂²（會塌）。
- `s3_separation_report.py`：定位表、曲面圖 fig14、附錄兩表。room 的讀法：固定面板自己的 w 切片、
  沿 λ 內插、T=30/120 各一值（(w,T,sep) 與 λ 綁在一起，二維內插會跨 regime）。

## 結果（對核心 claim 不利，照實入文）
| 面板 | λ⁽⁺⁾ | R² learned / tax | room（T=30 / 120） | 模擬 learned | 實現 |
|---|---|---|---|---|---|
| Online Retail | 0.553 | 0.44 / 0.17 | +0.29 / −0.11 | −2.5 / −1.5 | −0.05 |
| M5 | 0.841 | 0.79 / 0.30 | −0.63 / −0.35 | −1.9 / −1.1 | +0.58 |
| Auto | 0.940 | 0.82 / 0.14 | −0.85 / −0.21 | −1.2 / −0.3 | −0.31 |
| **Carparts** | 0.224 | 0.68 / 0.01 | **+1.82 / +2.45** | −1.0 / −0.9 | −0.08 |
| RAF | 0.917 | 0.95 / 0.09 | −0.81 / −0.13 | −1.4 / −0.3 | +0.03 |

- **OR 的 room 至多 0.3%**：「Online Retail is the decisive case」不成立，舊 +2.2%@λ≈0.5 是混分離度的假點。
- **唯一有 room 的是 Carparts（1.8–2.4%）**，實現 −0.08%，且模擬裡同一目標函數在同座標也輸（−0.9～−1.0%）
  → learnability 的主張只在 Carparts 成立，但多了模擬證據。
- 曲面通則：真 partition 的回報由遺忘 regime 決定不亞於 λ：w=0.90 多數分離度 1–2.7%；w=0.95 只在 λ<0.4 超過 1%；
  w=1 從不。learned mixture 在中等分離度幾乎全負（−1～−2.7%），只有 sep≥2 且 T=120 追上真 partition。
- 注意：sep=0 仍有 ~1% 增益（occurrence spread 0.8 固定），屬 occurrence block；附錄已註明。

## 稿件（`w16_separation_tex.py`，兩目錄，原文 ORIG 保留）
abstract / Intro / 貢獻(3) / Conclusion / Limitations(4) 的「OR 決定性案例、>2%」全部改為
「四面板無 room、Carparts 有 1.8–2.4% 但真實與模擬的 learned 都拿不到」；§5.5 加分離度句；
§5.6 兩段重寫（「Where the room is, and whether it is collected」）；fig10 caption 改為誠實說明灰線混分離度；
新 fig:separation（正文）、附錄段落 + tab:sepsim + tab:separation。v5_aistats 35 頁、0 錯誤。

## 檔案
`sep_surface.csv`、`sep_surface_cells.csv`、`real_separation.csv`、`real_separation_groups.csv`、
`separation_placement.csv`、`tab_separation.tex`、`tab_separation_sim.tex`、`s1/s2_run_meta.json`、
`figs_v3/fig14_separation_surface.{pdf,png}`。

## 第二版曲面（`s1b_separation_surface_v2.py`，占用 occurrence 縮放＋修法 arm；已取代第一版入稿）
- occurrence_spread = 0.8·min(sep,1)：sep=0 兩個 block 都無結構 → 真 partition 增益在噪音內（第一版的 ~1% 確為 occurrence 貢獻）。
- 真 partition 回報：w=0.90 從 sep 0.5 起 1–2.7%；w=0.95 只有 sep 0.5、T=120 超過 1%（λ≈0.4）；w=1 從不。
- **修法 arm 全部無效**（全網格平均）：learned −0.58%、+credibility −0.66%、+split −0.79%、兩者合併 −0.56%；
  Carparts 座標上 split+cred −1.1～−1.5%。判定：維持 negative，Limitation(4) 加一句。
- 定位（w 切片沿 λ 內插，T=30/120）：OR −0.5/+0.6、M5 −0.3/−0.5、Auto −0.5/−0.3、**Carparts +2.0/+1.7**、RAF −0.5/−0.4；
  結論與第一版相同：四面板無 room、Carparts 有 room 而 learned 拿不到。
- 稿件：`w16b` 已把數字換成第二版（四面板 −0.5～+0.6、Carparts 1.7–2.0、OR ≤0.6）、tab:sepsim 加 fix 欄、附錄註明第一版存檔取代。35 頁 0 錯誤。

## 收束方向（作者裁決中）：理論＋診斷排第一貢獻，以 locality/globality 事前判準立論；方法第二；評估警告第三。下一步開 v6 骨架（8 頁）。
