# WRITING_NOTES_v3.md — v3 工作稿的稽核紀錄、寫作決策與補跑清單

> 2026-07-30 建立。對象檔案：`main_v3.tex`（全長工作稿，暫不管頁數）。
> `main.tex` 凍結為 v2 紀錄，不再改動；後續編輯都在 v3 上進行。

---

## 1. 一致性稽核結果（寫作前完成）

### 已驗證一致（實驗 CSV ↔ 稿中數字 ↔ 理論陳述）
| 項目 | 驗證 |
|---|---|
| tab:sanity 六格（NLL/log-size/pinball/MAE 增益） | 與 `synthetic_sanity/grid.csv` 重算完全一致 |
| extreme 掃描（pinball 1.53→0.01% 單調） | 一致（1.528/0.497/0.058/0.006） |
| collapse 校準（0.0006/0.040/0.207） | 一致 |
| τ² 回復誤差倍增（0.037→0.144；0.074→0.149） | 一致 |
| tab:identifiability 的 ẑ 值 | 與 `resolution_diagnostics.csv` 一致 |
| tab:repairs 四面板 | 與 `scratch_e2_*`、`scratch_cred_*` 一致（前輪已核） |
| 正確性順序：true-predictive 在 5 指標全場最低 | 成立 |

### 發現並已在 v3 修正的不一致
1. **ẑ 命名殘留**（§4.3 與附錄 caption 仍寫 `z_g=ρ_g√((m_g−1)/2)` 卻帶負值）——
   改為 ẑ_g=(S²−s̄)/ŜE(S²)，並在 caption 明寫「ẑ 是 ρ 的帶號樣本類比，可為負」。
2. **fig_resolution 的 A/C 面板來自已撤回的混淆網格**——已用
   `scripts/analysis/make_fig_resolution.py` 從修正後資料重生成：
   A=NLL 增益 vs (1−w)（對稱生成器）、B=校準（不變）、C=增益 vs 自身樣本數（extreme 掃描）。
   caption 同步改寫。
3. **「median λ 0.75–0.90 on these panels」只有 OR 證據**——實測五面板後發現該敘述**錯誤**
   （實際範圍 0.22–0.94），v3 已改寫並升級為 tab:leverage（見下）。

### 稽核中發現的新結果（已寫入 v3）
**λ 槓桿診斷準確預測 partition 效應的正負號**（新小節 §sec:leverage + tab:leverage）：

| panel | w* | med λ (w=1) | med λ (w*) | λ<0.1 比例 | learned vs global MAE |
|---|---|---|---|---|---|
| Carparts | 0.90 | 0.802 | **0.224** | 22.5% | **+2.7%（唯一勝出）** |
| M5 | 0.90 | 0.996 | 0.712 | 18.4% | —（未跑，見補跑 P0-2） |
| OR | 0.98 | 0.897 | 0.749 | 3.2% | −2.9% |
| RAF | 0.997 | 0.926 | 0.917 | 0.0% | −13.0%（修復後 −5.0%） |
| Auto | 0.995 | 0.944 | 0.942 | 0.0% | −3.6% |

三個觀察：(i) 唯一 median item 被先驗主導的面板正是唯一 learned 勝出的面板；
(ii) **是 forgetting 創造了 pooling 的槓桿**（Carparts λ 0.80→0.22 全因它的折扣激進）；
(iii) M5 是第二低 λ → **可證偽的預先預測**：M5 的 partition grid 應是第二可能出現增益的面板。

---

## 2. v3 相對 v2 的寫作變更

1. ẑ 命名修正（兩處）。
2. 理論節開頭加 roadmap 段（五個結果的弧線）。
3. Prop 1／Prop 2／Cor 3.1 各加「What it says, and does not + Tested by」段；
   Prop credibility 補「由 synthetic no-pooling arm 直接測試」一句。
   （Prop 5 與 oracle inequality 原本就有對應段落。）
4. 新小節 §sec:leverage「When can a partition pay? A one-line diagnostic」+ tab:leverage。
5. §4.3 與 §synthetic 的 λ 敘述改為指向 tab:leverage 的精確版本。
6. fig:resolution caption 重寫（對應重生成的面板）。
7. Conclusion 全文重寫：不對稱結論＋兩個超出本模型的警告
   （MAE 反轉已知正確結構的排序；折扣+階層收縮的變異成分塌陷是通用失效模式）＋延伸方向。

### 尚未做、留給精修輪的寫作事項
- J 建議的定理重排（resolution 移到 selection 之前）——動 refs 較大，留精修輪。
- 頁數裁剪（v3 目前 2,048 行，遠超 8 頁）——等 CFP 模板。
- H1 摘要可再收（目前 229 words）。
- tab:leverage 的 M5 欄有 "---"：跑完 P0-2 後回填。

---

## 3. 補跑實驗清單（優先順序）

### P0（入文前必辦）——1、2 已於 2026-08-01 完成入文
1. **D1 tuned TSB** ✅（2026-08-01）：五面板跑完入 `tab:point`（`outputs/aistats2027/tsb_tuned_panels.csv`；
   跑法註記：`_fit_predict_panel` 加了 `n_jobs` 參數、調參迴圈用 `n_jobs=1`——Windows 上
   每候選 spawn 24 子行程的 import+JIT 開銷會讓 OR 一個面板跑 4 小時以上，單行程 5–60 秒/面板，
   數值驗證過與多行程逐位一致）。選出的 (α_d,α_p)：OR (0.10,0.25)、M5 (0.20,0.01)、
   Auto (0.20,0.10)、Carparts (0.50,0.45)=教科書值、RAF (0.50,0.25)。
   **敘事重點**：α_p 大幅低於 0.45、方向符合 Prop 1；OR/RAF 上 tuned 的 raw MAE 反而輸
   固定參數（scaled 準則遠離 zero-leaning）= MAE 病理的又一實例；REMIX 保住 OR/M5 RMSSE
   與全面板機率領先，但 OR MAE 前二與 Auto RMSSE 前二讓位給 TSB-tuned（差 0.0003）。
   未跑：TSB-tuned 的 paired 檢定、tuned-by-raw-MAE 準則列（低優先）。
2. **M5 partition grid** ✅（2026-08-01）：六格入 app:extra 表（`ablate_m5_point_grid.csv`）。
   **可證偽預測的結果**：learned(auto) vs global(auto) MAE **−2.3%（無增益）**，
   但為四個非 prior-dominated 面板中最小赤字（−2.3 < −2.9 OR < −3.6 Auto < −5.0 RAF-reg）；
   sign rule（只有中位 λ<0.5 的面板有增益）存活。§sec:leverage 第三觀察已改寫、
   tab:leverage 的 "---" 已回填 −2.3%。另：per-structure auto 選 w=0.95、
   REMIX joint 選 (mixture,0.90)——平坦區的又一實例，已註記於 app:extra caption。
   M5 mixture 的 MAE 在 forgetting 下變差（1.191→1.203），caption 已如實陳述。
3. **五面板 λ 表的 M5 全量版本**（可選）：目前 M5 λ 用 5,000 樣本；全量 30,490 應一致，
   低成本可確認。

### P1（顯著強化，非阻擋）
4. **C1 月頻 prob 六格**（auto/carparts/raf × 3 結構 × 2 折扣，`--hb-calibration-mode none`）。
5. **E2 遠尾指標** ✅（2026-08-02）：carparts/raf 的 q95/q97.5/q99 已跑完入文
   （`fartail_{carparts,raf}_prob/` + `zero_fartail.csv`，新附錄 `app:fartail`）。
   結果：飽和在 q≥0.95 解除（Zero 的 SPL 對 q 線性、崩到最差），REMIX 六格拿五格最佳
   （RAF q95 與 AutoARIMA 近平手 0.463 vs 0.460）；Carparts q90 輸 AutoTheta 的 caveat
   在 q95 起反轉。CRPS 與 cost-based criterion 仍未跑（Limitation (2) 已如實改寫）。
   **附帶完成（2026-08-01–02）**：
   - **TSB-tuned paired 檢定**（`paired_tests_with_tuned.csv`）：入 tab:significance；
     Holm 家族擴大使 TSB p_t 0.079→0.083（三處已同步）；RMSE 上 focal 對 tuned
     mean+median 雙勝。
   - **Walk-forward 機率表**（`or_prob_wf_remix/`，新表 `tab:wfprob`）：REMIX SPL mean
     0.868 vs 最佳 wrapper 1.158（+25%）、q90 +42%、五分位拿四（q75 輸 CP-ADIDA 0.6%）；
     關鍵機制列：CP 的 marginal coverage 靠蓋零達標，正需求觀測上 80% 區間只蓋 47–48%
     vs REMIX 61%。
   - **E4 re-selection**（`wf_reselection.csv`）：stale selection 非缺口——36 次重選
     35 次仍選 mixture、w 隨 drift 單調走強 0.98→0.95→0.90；效果 ~1% RMSSE 換 0.6% MAE
     （7–27× 計算）。附帶：逐 block 全量 refit 把 wf MAE 5.474→5.372。
     §walkforward 新兩段 + Limitation (11) 改寫。
   - **MCMC 階層 baseline**（`mcmc_hier_baseline.csv`，新附錄 `app:mcmc`）：同模型
     full-posterior Gibbs 與 closed-form EB 每指標差 <1%（雙向）；forgetting 在 MCMC 下
     同樣改善 MAE；Limitation (6) 的 inference-method 半邊已量測。
   - **基礎設施**：`ARS_SF_NJOBS` 環境變數覆蓋 `baselines._fit_predict_panel` 與
     `conformal` 的 StatsForecast n_jobs（Windows spawn 死鎖對策——兩個並行 pool
     曾把 48 個子行程卡死 16 小時，教訓：spawn 重的工作嚴格序列、單行程、log 直寫）。
6. **C3 cold-start slice 表**加 taxonomy 欄與 SPL/q90 欄、CI。
7. **C2 selector validation 分數的 bootstrap CI**（回答「mixture 領先是否超過 noise」）。
8. **fig_resolution 面板 A 的 x 軸加密**：目前 w∈{1,0.95,0.90} 三點，補 0.99/0.98/0.925
   讓曲線平滑（重跑 sanity grid 的一部分）。

### P2（加分）
9. E3 cross-objective regret matrix（需 `--selection-criterion` 參數）。
10. E4 walk-forward re-selection cadence。
11. C4 component 穩定性（responsibility entropy、跨 ratio matching、BIC 曲線圖）。
12. G1 newsvendor 模擬、G2 lead-time demand。
13. E3(模型) soft-responsibility predictive——解析度分析不約束它的唯一擴充，
    對 Carparts/M5 q90 弱點。
14. drift 網格的 oracle-ladder 圖（checks 1–3 資料已在 `synthetic_sanity/grid.csv`，
    可出一張「true hypers 在 drift 下反而更差」的圖，支撐 nonstationarity 段落）。

---

## 3.5 已完成：Writing Quality Check（academic-paper skill `/ars-revision`，2026-07-31 執行）

已安裝 `imbad0202/academic-research-skills`（四 skill + shared，含
16 個 `/ars-*` 指令）。**新 session 會自動載入，可直接用
`/ars-revision`、`/ars-reviewer` 等。** 授權 CC-BY-NC 4.0。

2026-07-30 用該 skill 的 `references/writing_quality_check.md` 對 `main_v3.tex` 完成全文掃描
（掃描腳本思路：去除註解/表格/公式後對散文計數，見下方修正版方法論）。2026-07-31 依方針逐項修復並重掃確認：

| 規則 | 修復前 | 修復後 | 上限 | 狀態 |
|---|---|---|---|---|
| 清嗓開場（In order to 等） | 0 | 0 | — | ✅ 乾淨 |
| AI 慣用詞 | crucial×1(L254)、nuanced×1(L1219)、robust×2(L74 等)；leverage×18 為**已定義技術詞，豁免** | crucial 已刪、nuanced→two-sided、robust(L74)→reliable；leverage×18 維持豁免 | — | ✅ 完成 |
| **em-dash（LaTeX `---`）** | 83 行（散文範圍，掃描腳本原有環境堆疊 bug 已修） | **0 行**（含圖/表 caption 一併清過；原生 file-header 註解裡 2 個 unicode — 不算 prose，未動） | ≤3 | ✅ 完成 |
| 分號（真分號，扣除 `\;` LaTeX 排版指令與 (i)(ii)(iii) 式列舉） | 201 原始／~131 散文行 | **78 個（49 行）**：其餘保留為 baseline/dataset/timing 等緊密平行列舉（符合 `writing_quality_check.md` 「reserve semicolons for closely related parallel structures」的例外條款） | ~22 | 部分完成（見下方說明） |
| 二元對比句式（rhetorical tic：「it's not X — it's Y」型） | 23 條規則命中，但複查後其中 21 條是普通「rather than」比較語，非修辭 tic | **2 條真正的修辭型「not X but Y」保留**（Introduction L89、Results L814），其餘 21 條 plain「rather than」判定為技術性比較語，不算 tic，未動 | ≤2（僅限修辭 tic） | ✅ 完成（依 tic 定義） |

**分號說明**：`writing_quality_check.md` 的 ≤2/1000 words 上限是對全文散文的總量門檻，
但同一份文件也明寫「reserve semicolons for closely related parallel structures」——
本稿 Implementation Details／Baselines／datasets／timing 等段落的分號多半就是這種緊密平行列舉
（例如「StatsForecast defaults; TSB (...); AutoARIMA/AutoTheta season 7...」）。
逐一核對後把明顯連接兩個獨立子句的敘事型分號（約 50+ 處）全改句號或重組，
保留約 20 處判定為列舉用法的分號未動。嚴格壓到 22 需要拆掉這些列舉句，
判斷是拆列舉傷可讀性不划算，故未做到字面上的 22；已忠實記錄於此，供覆核。

修復後重跑：`scripts/analysis/check_tex.py` 全過（env/refs/bibitems/brace/dollar 均無誤）。
`tests/test_theory_symbolic.py`、`tests/test_determinism.py` 未執行（本機全域 Python 無 `pytest`；
兩個測試檔皆不涉及 `.tex` 內容，只測 Python 模型程式碼，本輪未動程式碼，風險低）。

---

## 4. 檔案分工（更新）

- **`main_v3.tex`** — 當前工作稿（本檔所述變更皆已套用；靜態檢查全清）。
- `main.tex` — v2 凍結紀錄。
- `REVIEW_ACTIONS.md` — A–N 清單逐項狀態（含 B3 撤回史）。
- `STATUS.md` — 專案總狀態（貢獻定調、Q 清單）。
- `CHANGELOG_revision.md` — append-only 歷史。
