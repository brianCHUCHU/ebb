# SUMMARY — 2026-09-06（walk-forward 五面板收官、TweedieGP wf、per-item w_i 閘門）

> 分工：數據與排版由我，論證文字由作者。失效敘述以 `% [2026-09-06 STALE]`
> 註解標在 .tex 內（含新數字），措辭未動。ACI 折衷（作者裁決）已落實：
> ACI-ADIDA 列保留、20% 主張保留、「adaptive conformal 在 M5 領先」一句放在
> tab:wfprob caption（事實敘述）。

## 任務狀態

| 項 | 狀態 |
|---|---|
| W1 TweedieGP wf：Carparts / Auto / RAF / Online Retail | ✅ 全部完成（released defaults；monthly 每 block refit、OR 每 4 block；12 workers）。牆鐘 45 / 36 / 141 / 101 分 |
| W3 五面板 wf 彙整 | ✅ `wf_all_panels.csv`、`wf_all_panels_wide.csv` |
| W4 LaTeX 列產生 | ✅ `tab_wfprob_main.tex`、`tab_wf_full_<panel>.tex`、`wf_standings.csv` |
| W5 稿件連動（兩目錄） | ✅ tab:wfprob 改五面板（Mean/q90）；附錄五張全表（q10–q90、Mean、Cov@80、Cov⁺@80、AIW）；STALE 註解 abstract/Intro/Conclusion/§5.3/附錄段 |
| tab:efficiency 壞列修復 | ✅ TweedieGP 列尾單反斜線（09-05 遺留）→ 兩目錄修正；此前 PDF 雖有輸出但 log 有 10 個錯誤 |
| 重編譯 | ✅ v5_aistats 27 頁、0 錯誤、0 未解引用、overfull 1 個（5.1pt <10pt）；v5 29 頁、overfull 5 個皆為既有 |
| P1 per-item w_i 閘門（五面板） | ✅ 跑完；**閘門未過**（見下） |

## ★ Headline：walk-forward 五面板最終排名（mean SPL，非退化排名）

| 面板 | 第一 | 第二 | EBB 對最強對手 |
|---|---|---|---|
| Online Retail | **TweedieGP 0.8526** | EBB 0.8540 | **−0.16%**（vs ACI 0.9108 仍 +6.2%） |
| Carparts | **TweedieGP 0.3121** | EBB 0.3145 | **−0.8%**（vs ACI 0.3417 +8.0%） |
| Auto | **EBB 0.2910** | TweedieGP 0.2924 | **+0.5%**（vs AutoARIMA 0.3139 +7.3%） |
| RAF | **EBB 0.2681** | TweedieGP 0.2801 | **+4.3%**；Zero 0.2669 低於所有方法（diagnostic 列） |
| M5 | **ACI-ADIDA 1.4192** | EBB 1.4851 | **−4.4%**（vs 最佳靜態 CP-IMAPA 1.5457 +3.9%） |

**對 EBB 不利（預先聲明過，照實入表）**：wf 下 EBB 只在 Auto、RAF 領先，
與 fixed-origin 相同（兩面板）；「wf 全面領先」不成立。TweedieGP 在 wf 也拿走
OR 與 Carparts（含兩者 q90：OR 1.3791 vs 1.4002、Carparts 0.4326 vs 0.4578），
ACI 拿走 M5。EBB 五面板全部前二。q90：EBB 只在 RAF 第一（0.4864）。

有利的一半：EBB 對所有非 TweedieGP 對手五面板全勝（OR +6.2%、Carparts +8.0%、
Auto +7.3%、RAF +13.1% vs ACI）除 M5（ACI −4.4%）；對全部靜態 CP wrapper 五面板全勝。
Auto/RAF 的 wf 領先幅度（0.5%/4.3%）與 fixed-origin（1.6%/3.0%）同量級。

給作者的 STALE 對照（tab:wfprob 段落原句 → 新事實）：
- 「+6.2% over ACI-ADIDA on OR」仍成立；「q90 +5.6% over AutoTheta」對象已非最強（TweedieGP q90 更低）。
- 「+9.9% over CP-IMAPA on Carparts, wins all five quantiles」→ CP-IMAPA 現為第三；EBB 保 q10–q75、失 q90 與 mean。
- 「leads under strict walk-forward on both a long daily and a short monthly panel」→ 只在 Auto、RAF 領先。
- 21% ACI-over-static（1.1576→0.9108）不變；coverage/sharpness 分離段（0.646 vs 0.621、19.67 vs 10.31）不變。

TweedieGP wf 的 coverage 側（附錄全表）：OR Cov⁺ 0.709 / AIW 13.03（EBB 0.621 / 10.31）；
Carparts 0.662 / 1.46（EBB 0.669 / 1.42）；Auto 0.828 / 8.81（EBB 0.873 / 9.17）；
RAF 0.198 / 4.07（EBB 0.041 / 0.95）。

## P1：per-item forgetting rate w_i 閘門判定 —— **未過，不採用（negative result）**

`peritem_discount_eval.csv`（fixed-origin、κ 在外層 tail 選、外部 B=20）：

| 面板 | scalar w* | κ*（tail 選） | per-item mean SPL | 變化 | TweedieGP |
|---|---|---|---|---|---|
| Online Retail | 0.8849 | 10 | 0.8846 | −0.03% | 0.8845 |
| M5 | 1.6707 | 3 | **1.6515** | **−1.15%** | 1.6228 |
| Auto | 0.2937 | 30 | 0.2940 | +0.10% | 0.2985 |
| Carparts | 0.3229 | ∞ | 0.3229 | 0（退回純量） | 0.3214 |
| RAF | 0.2680 | 100 | 0.2678 | −0.07% | 0.2763 |

對照 `docs/DESIGN_per_item_discount.md` §4 四條判準：
1. OR 或 M5 改善 ≥1% 且 Auto/RAF 不惡化 >0.3% → **過**（M5 −1.15%；Auto +0.10%、RAF −0.07%）。
2. 對 TweedieGP：收復 OR（需 <0.8845）或 M5 差距 <1.5% → **不過**（OR 0.8846 差 0.01%；M5 差距 1.77%）。
3. κ 曲線內點最優 → **過**（M5 tail 分數 κ=3 最低，κ=0 崩到 1.8155、κ=∞ 1.6751）。
4. 選模時間 ≤3× → **過**（M5 內層曲線 140 s、整面板 15 分）。

判準 2 未達 → 依設計文件寫成 negative result（"per-item forgetting narrows but does not
close the M5 gap"）。附帶觀察：M5 有 49% 品項選了比 0.95 更快的遺忘（w 中位 0.95、
κ=3）；OR 只有 19% 且改善可忽略；κ=0（無收縮）在 M5 崩壞證明收縮必要。
**若作者要改判採用**：需 §5 的全套連動（wf 五面板重跑、selection surfaces、λ 表語義、
fig10、顯著性、效率表、三 seed、§3.4 文字、Prop 2 附註），估 2 天機器＋2 天稿；
且 M5 仍輸 TweedieGP 1.77%，並不改變任何面板的第一名。`src/models/tsb_hb.py` 的
per-item 支援保留（純量路徑 sha 位元不變已驗證），不影響現有結果。

## 檔案

- `wf_tweediegp_{carparts,auto,raf,online_retail}.csv` + `_quantiles_` + `w1_run_meta_*.json`
- `wf_all_panels.csv`、`wf_all_panels_wide.csv`（W3）；`wf_standings.csv`、`tab_*.tex`、`w4_run_meta.json`（W4）
- `peritem_discount_eval.csv`、`peritem_w_<panel>.csv`、`p1_run_meta.json`（P1）
- `wf_tmp/`（402 MB TweedieGP 每次 refit 的 in/out CSV；可刪，未刪）
- 腳本：`scripts/integrity/w3_assemble_wf_tables.py`、`w4_wf_latex_tables.py`、`w5_patch_wf_tex.py`、`w5b_fix_efficiency_row.py`

## 待作者裁決

1. tab:wfprob 段落、abstract/Intro/Conclusion 的 wf 敘述改寫（STALE 註解內有全部數字）。
2. 是否接受 per-item w_i 的 negative-result 判定（或改判採用，代價見上）。
3. 投稿定位：fixed-origin 與 wf 兩協定下 EBB 皆為「兩面板第一、五面板前二、
   對非 GP 對手全勝、成本 6.5–80×」；「準確度最一致」敘事已無法支撐。
4. 仍未 commit：paper_v2/v5*、scripts/integrity/、outputs/integrity_*、outputs/2026-09-0[56]。

## 定位改寫（作者指示「先按照你的想法來改」，2026-09-06 晚）

新定位：**一致性＋成本前沿、理論預測的負面結果**。改動全部在兩稿目錄同步，
每段被替換的原文以 `% [ORIG 2026-09-06]` + `% ORIG|` 逐行保留在 .tex 內，可直接還原。
所有 `STALE` 標記已清空（8 個 ORIG 區塊）。

### 新證據（`w6_regret_and_wfcost.py`）
- **Max regret 表**（`regret_fixed.csv`、`regret_wf.csv`、`tab:regret` 正文、`tab:regret-full` 附錄）：
  對各面板非退化最佳者的 mean SPL regret。Fixed：EBB 2.95%（M5）、TweedieGP 3.10%（RAF）；
  wf：EBB 4.64%（M5）、TweedieGP 4.48%（RAF，4 面板）。**只有這兩個方法五面板全在 5% 內**；
  其餘每個方法最差面板 ≥15%（ACI 15.1%、CP-ADIDA 27–36%、AutoTheta 53–61%、Tweedie-GLM 151%）。
- **wf 牆鐘表**（`wf_cost.csv`、`tab:wfcost` 附錄 app:efficiency）：EBB 五面板 wf 全程
  OR 9.7s / Carparts 1.5s / Auto 1.8s / RAF 6.7s / M5 36.6s（單核、含初始化；四小面板重新計時
  `ebb_wf_timing.csv`，M5 取 h4 元資料）；TweedieGP 12 workers 2139s（Auto）～8484s（RAF）、
  OR 6030s；M5 投影 51h（每 4 block refit、24 次）／197h（每 block）。
- **負面結果附錄 app:negative**（新小節，含 `tab:shape` 預測律形狀五面板、`tab:peritem` 逐品項 w_i 五面板＋閘門敘述）。

### 改寫的段落（新文字為我的草稿，作者可全改）
abstract 實證句、Intro「empirical position」段、貢獻 (1)(3) 各加一句、§5.1 comparators/protocol 句、
tab:prob caption、§5.2 首段、§5.3 tab:wfprob 後首段（＋插入 tab:regret）、coverage/sharpness 句補附錄引用、
Conclusion 首段、Limitations (1) 整段、(4) 一句、附錄 E.3 wf 段。

### 編譯
v5_aistats 30 頁、0 錯誤、0 未解引用、overfull 1 個 5.1pt（既有）；v5 30 頁、overfull 6（5 既有＋1 個 1.1pt）。
tab:regret 第一版超寬 36pt → footnotesize＋tabcolsep 2.6pt＋Carparts 縮寫 Carp. 後歸零。

### 遺留
- `figs/horizon_decomp.pdf`（附錄 Fig. 5）圖例仍寫 "REMIX (ours)"，產生器 `make_figures_v3.py fig07`
  讀的是 `outputs/aistats2027/`（作廢目錄），重出需以修正配置重跑 OR horizon 分解——未動，待裁決。
- 正文仍約 12 頁（AISTATS 上限 8 頁正文），裁減由作者。
