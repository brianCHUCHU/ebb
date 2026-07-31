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

### P0（入文前必辦）
1. **D1 tuned TSB**：程式已在 `src/models/tsb_tuned.py`，OR 背景跑兩次都沒回收到輸出，
   需重啟並確認（`fit_predict_tsb_tuned`，25 候選、同一 80/20 切分）。五面板入 tab:point。
   ——公平性問題，審稿人第一個問的。
2. **M5 partition grid**（global/taxonomy/mixture × w∈{1,auto}，point）：
   回填 tab:leverage 的 "---"，並驗證 λ 診斷的預先預測。這是 v3 新增的可證偽宣稱，
   跑出來無論方向都是內容。
3. **五面板 λ 表的 M5 全量版本**（可選）：目前 M5 λ 用 5,000 樣本；全量 30,490 應一致，
   低成本可確認。

### P1（顯著強化，非阻擋）
4. **C1 月頻 prob 六格**（auto/carparts/raf × 3 結構 × 2 折扣，`--hb-calibration-mode none`）。
5. **E2 遠尾指標**：q95/q97.5/q99 + CRPS，至少 RAF（回應 saturation caveat）。
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

## 3.5 進行中：Writing Quality Check（academic-paper skill，2026-07-30 中斷待續）

已安裝 `imbad0202/academic-research-skills` 至 `~/.claude/skills/`（四 skill + shared）與
`~/.claude/commands/`（16 個 `/ars-*` 指令）。**新 session 會自動載入，可直接用
`/ars-revision`、`/ars-reviewer` 等。** 授權 CC-BY-NC 4.0。

已用該 skill 的 `references/writing_quality_check.md` 對 `main_v3.tex` 完成全文掃描
（掃描腳本思路：去除註解/表格/公式後對散文計數）。**掃描結果（待修）**：

| 規則 | 現況 | 上限 | 狀態 |
|---|---|---|---|
| 清嗓開場（In order to 等） | 0 | — | ✅ 乾淨 |
| AI 慣用詞 | crucial×1(L254)、nuanced×1(L1219)、robust×2(L74 等)；leverage×18 為**已定義技術詞，豁免** | — | 小修 |
| **em-dash（---）** | **83 行（~60 個結構）** | ≤3 | ❌ 主要工作 |
| 分號 | ~136（11,114 words） | ~22 | ❌ 需大減 |
| 二元對比句式（not X but Y / rather than） | 23 | ≤2（針對修辭 tic） | 需減半 |

**修復方針（已規劃、未執行）**：em-dash 逐一改為逗號對／括號／冒號／分句
（範圍語 en-dash `--` 如 0.75--0.90、Bühlmann--Straub 為正確用法，不動）；
分號集中在我起草的敘事段落改句號；「not because...but because」全文留 1 處（結論），
其餘改寫；crucial→刪、nuanced→two-sided、robust(L74)→reliable。
修完重跑掃描 + `check_tex.py` + 兩個測試檔，再 commit。

---

## 4. 檔案分工（更新）

- **`main_v3.tex`** — 當前工作稿（本檔所述變更皆已套用；靜態檢查全清）。
- `main.tex` — v2 凍結紀錄。
- `REVIEW_ACTIONS.md` — A–N 清單逐項狀態（含 B3 撤回史）。
- `STATUS.md` — 專案總狀態（貢獻定調、Q 清單）。
- `CHANGELOG_revision.md` — append-only 歷史。
