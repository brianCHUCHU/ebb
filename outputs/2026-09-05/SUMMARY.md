# SUMMARY — 2026-09-05（任務 2 放行後：TweedieGP OR/M5 全量、效率表重出、改名 EBB）

> 分工：數據與排版由我，論證文字由作者。所有現在失效的敘述以
> `% [2026-09-05 STALE]` 註解標在 .tex 內，措辭未動。

## 任務狀態

| 項 | 狀態 |
|---|---|
| 改名 REMIX → EBB | ✅ 兩稿目錄全換（65 處呼叫、標題、running head）；命名 footnote 為草稿待作者定稿 |
| 2 TweedieGP OR 全量 | ✅ 3,649 條、29 分、0 失敗（released defaults；T≤200 → 全點 inducing） |
| 2 TweedieGP M5 全量 | ✅ 5,000 條、127 分、0 失敗（m=200/'log'；293 條零訓練窗→零預測 fallback，與 t3 慣例同） |
| 3 效率表 TweedieGP 列重測（同 runner、12 workers） | ✅ Carparts 302.1s、0.120 s/序列、單執行緒中位 1.51s；加 OR/M5 列；三項註記入 CSV notes 與附錄段 |
| 3b fig11 重畫 | ✅ 以新 Carparts 成本重畫（EBB 標籤） |
| 1f/1e 統一評分＋顯著性（含 OR/M5 TweedieGP） | ✅ `outputs/integrity_2026-09-05/`（f57 舊命名） |
| 4 M5 walk-forward 方案 c | 🟡 重跑中（block 2 外推 3.5h） |
| 5 連動 | 🟡 表格側完成；prose 側標 STALE 等作者 |

## ★ Headline 影響（對 EBB 不利，預先聲明過，照實入文）

TweedieGP（同一評分函式重算）：

| 面板 | TweedieGP mean / q90 | EBB mean / q90 | 差 |
|---|---|---|---|
| Online Retail | **0.8845** / **1.5218** | 0.8849 / 1.5268 | TG 領先 0.05%（顯著性未定，t p=0.71） |
| M5 | **1.6228** / 2.1723 | 1.6707 / 2.4758 | **TG 領先 2.9%（顯著，p=0.003/0.035）**；q90 仍 CP-Croston 2.1551 最佳、TG 第二 |
| Auto | 0.2985 / 0.2598 | **0.2937** / 0.2599 | EBB +1.6%（顯著） |
| Carparts | **0.3214** / **0.4600** | 0.3229 / 0.4817 | TG +0.5%（未定） |
| RAF | 0.2763 / 0.5219 | **0.2680** / **0.4856** | EBB +3.0%（顯著） |

**「五面板中四面板 mean SPL 最低」→「五面板中兩面板（Auto、RAF）」**；
TweedieGP 在 OR/M5/Carparts 領先。顯著性家族 48→50 對：**43 顯著較好、
1 顯著較差（M5 對 TweedieGP）、6 未定**（M5 四 CP、Carparts-TG、OR-TG）。

給作者措辭用的數字：EBB 對最強對手的 margin：Auto +1.6%（vs TG）、
RAF +3.0%（vs TG）；落後：OR −0.05%、M5 −2.9%、Carparts −0.5%（皆 vs TG）。
對非 TweedieGP 對手，EBB 仍五面板全最佳（OR 對 iETS +10.1%、M5 對
CP-Croston +1.2%）。TweedieGP OR/M5 point：MAE 5.6669 / 1.2464（EBB 5.5325 /
1.2032 較佳）、RMSSE 4.7886 / 2.2763（TG 較佳）。遠尾（OR q95/97.5/99）
1.4921/1.3517/1.1315；（M5）1.5603/1.0557/0.6054——app:fartail 只涵蓋
Carparts/RAF，未入表。

## 已做的稿件連動（兩目錄同步）

- tab:prob：TweedieGP OR/M5 格填入，粗體重排（OR mean/q90、M5 mean 的
  best 移給 TG，EBB 降第二；M5 q90 第二給 TG；iETS/AutoTheta/CP-Croston/
  CP-SBA 的底線移除）。
- tab:spl-significance：OR/M5 family 10；M5 worse 欄 "1 (TweedieGP)"；
  undecided 加 TweedieGP；計數句改 43/50、1 worse、6 undecided。
- app:efficiency：M5「Cholesky NaN」段作廢，換成修正配置下的事實
  （OR 中位 5.1s、M5 17.3s、其中約半為 h=647 的 predict）＋作者指定三
  註記（同機同 session／0.24 vs 1.89 vs 0.97 與 train_longer／協定差異）；
  tab:efficiency TweedieGP 列＝302.1s、0.120（12 workers）。
- **STALE 註解（措辭待作者）**：abstract「lowest ... on four」＋「ahead on
  Carparts by half a percent」、Intro「on four ... margins from roughly one
  percent to ten」、§5.2 首段（four of five、10.1%/3.0%/1.6%/1.2%、
  「most consistent aggregate performer」）、Conclusion 首段、Contributions
  無數字不影響、setup 與 tab:prob caption 的「three monthly panels」
  （後者你說兩格補齊可整句刪，我未刪、回報狀態）。

## 效率表（`efficiency_carparts.csv`，今日目錄）

TweedieGP Carparts 由 253.2s@16w（舊 t3 runner、deterministic on）改為
302.1s@12w（同 h2 runner）；OR 1730.1s（0.474 s/序列）、M5 7617.9s
（1.524 s/序列）加列。牆鐘口徑下 EBB 對 TweedieGP 仍約 6.5×，單執行緒
口徑（0.019 vs 1.51）約 80×——兩口徑都在表。

## 對 EBB 不利的發現（照實列）

1. **TweedieGP 拿走 OR 與 M5 的 mean SPL**，四面板最佳縮成兩面板；M5 差距
   2.9% 且顯著。這是本輪最重的一條。
2. OR 上兩者統計不可分（0.05%），但 headline「10.1% margin」不再對
   最強對手成立（只對 iETS 成立）。
3. 效率上 EBB 對 TweedieGP 的牆鐘優勢僅 ~6.5×（平行攤提後）。

## 待作者裁決

1. abstract/Intro/§5.2/Conclusion 的重寫（數字在上）；投稿定位是否轉為
   「診斷＋負面結果＋效率」主軸（TweedieGP 在三面板領先後，「準確度最一致」
   的說法已無法支撐）。
2. tab:prob caption 的 TweedieGP 佔位句是否整句刪除。
3. 命名 footnote 定稿。
4. M5 wf（任務 4）約 3 小時後出，屆時補 tab:wfprob M5 欄。

## 階段 1 補完：預測律形狀變體（五面板，`predictive_law_variants_all.csv`）

同一組 EBB 後驗（單次擬合、無 B=20 平均），只換正值部分的分布形狀：

| 面板 | log-normal | log-t ν=10 | log-t ν=5 | Gamma（動差對齊） | σ×1.2 | TweedieGP |
|---|---|---|---|---|---|---|
| OR | 0.8849 | 0.8849 | 0.8850 | 0.8878 | 0.8853 | 0.8845 |
| M5 | 1.6730 | 1.6735 | 1.6744 | 1.6735 | 1.6750 | **1.6228** |
| Auto | 0.2937 | 0.2936 | 0.2940 | 0.2937 | 0.2954 | 0.2985 |
| Carparts | 0.3230 | 0.3228 | 0.3226 | 0.3227 | 0.3224 | **0.3214** |
| RAF | 0.2681 | 0.2680 | 0.2678 | 0.2673 | 0.2679 | 0.2763 |

**結論**：所有變體與 log-normal 的差距 ≤0.003，M5 上更厚或更薄的尾都略差；
TweedieGP 在 M5 的 2.9% 領先與 Carparts 的 0.5% 都不是形狀能補的。
差距來源在後驗（per-series 時變適應、單一潛變數同管零與正值），
Gamma/Tweedie size block 路線關閉；M5 病灶在 occurrence block（q90 歸零列
20.6% vs TG 13.4%），列為 Limitations／future work。

## 任務 4 完成：M5 walk-forward 方案 c（`m5_walkforward.csv`，93 blocks，全量 5,000，ARS_SF_NJOBS=1 後 2.3h）

| 模型 | q10 | q25 | q50 | q75 | q90 | **Mean** | Cov@80 | Cov⁺@80 | AIW |
|---|---|---|---|---|---|---|---|---|---|
| CP-Croston | 0.502 | 1.150 | 1.917 | 2.343 | 1.923 | 1.567 | 0.812 | 0.679 | 2.26 |
| CP-SBA | 0.495 | 1.137 | 1.909 | 2.385 | 1.956 | 1.576 | 0.812 | 0.684 | 2.28 |
| CP-TSB | 0.531 | 1.263 | 2.094 | 2.377 | 2.042 | 1.662 | 0.831 | 0.685 | 2.37 |
| CP-ADIDA | 0.479 | 1.108 | 1.889 | 2.330 | 1.963 | 1.554 | 0.827 | 0.690 | 2.28 |
| CP-IMAPA | 0.480 | 1.110 | 1.889 | 2.301 | 1.949 | 1.546 | 0.828 | 0.690 | 2.28 |
| **ACI-ADIDA** | 0.402 | **0.960** | **1.717** | **2.211** | **1.806** | **1.419** | 0.883 | 0.718 | 3.48 |
| Zero | 0.392 | 0.979 | 1.959 | 2.938 | 3.526 | 1.959 | 0.606 | 0.000 | 0.00 |
| EBB (mixture, 0.95, online) | **0.391** | 0.967 | 1.852 | 2.336 | 1.880 | 1.485 | 0.875 | 0.690 | 2.77 |

**對 EBB 不利（預先承諾入文）**：M5 wf 由 **ACI-ADIDA 領先**（mean 1.419 vs 1.485，
差 4.4%；q90 1.806 vs 1.880，3.9%），q25–q90 四個分位皆勝 EBB；EBB 只在 q10
（零飽和）最佳。有利的一半：EBB 勝過全部五個靜態 CP wrapper（對最佳 CP-IMAPA
+3.9%）與 Zero；**fixed-origin 的 q90 落後（2.476 vs CP-Croston 2.155）在 wf 下
翻正**（1.880 vs 1.923）——資料過時是 M5 q90 問題的一大部分，但 ACI 的自適應
把這一段也吃掉了。wf 相對 fixed-origin 的改善：EBB −11.1%（1.671→1.485）。
TweedieGP 在 M5 wf 不可行（~180h），表中標註。

Carparts ACI-ADIDA 補跑：mean **0.3417**（q90 0.4767）——成為 Carparts wf 最強
對手（原 CP-IMAPA 0.3492），EBB 0.3145 的領先由 9.9% 改為 **8.0%**；
tab:wfprob 的 Carparts ACI 格由 --- 改填。
