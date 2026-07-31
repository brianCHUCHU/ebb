# STATUS.md — REMIX / AISTATS 2027 單一權威狀態表

> 最後更新：2026-07-29。這份檔案是**目前唯一的權威狀態來源**。
> 其他檔案的分工：`CHANGELOG_revision.md` 記「做過什麼」（append-only 歷史），
> `QUESTIONS.md` 記「等作者裁決什麼」，`NUMBERS*.md` 記「每個數字的來源」，
> `SUBMISSION_CHECKLIST.md` 記「投稿前的機械檢查」，本檔記「現在站在哪、下一步做什麼」。
> `REMIX_revision_todolist.md`（同目錄）是 2026-07 那份原始稽核清單，保留作對照。

---

## 1. 三個貢獻（2026-07-29 重新定調）

前一版的貢獻 (2) 寫「理論解釋兩個軸為互補」，而 (3) 暗示 learned pooling 帶來增益。
Round 7 的三次獨立證偽（見 §2）顯示後者站不住。以下是與證據對齊後的版本。

### (1) 一個把古典方法當角落的模型平面
間歇需求預測有兩個幾乎總是被寫死的決策：**怎麼跨品項匯集資訊**（固定 ADI/CV² 分類，
或單一全域先驗）與**怎麼跨時間加權資訊**（均勻，或手調平滑常數）。REMIX 把兩者變成一個
可搜尋的平面，並保持閉式、決定性、O(N)。
- **Prop 1**：折扣後的後驗平均在自洽初始化下**精確等於** TSB 遞推，否則幾何收斂——
  50 年的產業啟發式是這個平面的無資訊先驗角落。
- 這條不受 Round 7 影響，是最穩的貢獻。

### (2) 支撐選擇器的理論，含一個可學習性的上限
- **Prop 3 + Cor 1**：漂移下的有限樣本 MSE 界給出 O(δ^{2/3}) 追蹤率，且漂移趨零時
  w*→1（sympy 符號驗證通過，`tests/test_theory_symbolic.py`）。
- **Prop 4**：有限候選集（8 個 w × 3 種 partition = 24）上以時序 hold-out 選擇的
  oracle 不等式，解釋 §4.3 觀察到的「平面在最優附近很平」。
- **待新增（建議，見 §5 項目 T9）**：Prop 2 目前是**單邊**的——它證明
  λ ≤ (1+(1−w)φ)⁻¹ < 1，即「先驗永不被洗掉」；但 Round 7 發現 λ 也會**塌到 0**，
  即「品項自身資料被完全洗掉」。折扣壓縮有效樣本數同時造成這兩件事。補上雙邊界
  （λ 有正下限，需要 τ² 有下界）會把負面結果變成理論陳述：**折扣能學到多少結構有上限，
  而且這個上限可以寫下來**。這是目前可得的最高價值理論補強。

### (3) 把兩個軸分開量測，並誠實報告哪一個有效
五個公開面板（Online Retail, M5, Auto, Carparts, RAF；19,158 條評估序列）：
- **機率端全勝**：mean scaled pinball loss 五個資料集全部第一，對手涵蓋古典、conformal、
  狀態空間（iETS）、Tweedie。
- **forgetting 軸是決定性的**：OR MAE 5.7623 → 5.5325（每種 pooling 下都改善）、
  區間銳化 12%（AIW 10.51 → 9.25）且 coverage 更貼近 nominal；M5 RMSSE 由落後翻為全場最佳。
- **partition 的選擇是二階的，而且我們證明了這不是實作缺陷**（三次獨立證偽，§2）。
  learned partition 的價值在於**移除手設閾值**且無總體代價，加上最稀疏 cold-start
  切片的 −4.5% MAE。
- **量測工具本身的病理**：全零預測在 5 個面板中的 3 個取得最低 MAE、3 個取得最佳 MASE。
  這使既有文獻大量基於 MAE/MASE 的排名失去意義，也解釋了為何我們以 SPL 為主指標。

**一句話定位**：我們不主張「學到的分群更準」；我們主張「這兩個決策應該從資料選，
而選擇器的行為有理論解釋，且在誠實的量測下 forgetting 軸帶來全部一階增益」。

---

## 2. Round 7 的核心結果：三次獨立證偽（論文需要寫進去）

問題：learned partition 總是被選中，但在 ablation 上不占優（OR MAE 5.6925 vs global 5.5325）。
三個假設，全部實作、量測、證偽：

| # | 假設 | 實作 | 結果 |
|---|---|---|---|
| i | post-selection 依賴（同一份資料既分群又估 hyper） | `--hb-label-split parity`（交錯切分，保留折扣輪廓） | MAE 5.6925 → 5.6264（−1.2%）、SPL 0.8869 → 0.8848。**不足** |
| ii | 變異成分邊界退化（τ² 塌陷） | `--hb-hyper-shrink credibility`（Bühlmann，無可調參數） | 塌陷完全消除（k_eff 上限 917,811 → 10.8；λ≈0 序列 37.7% → 0.7%），**但總體指標變動 <0.3%** |
| iii | 群 hyper 過強、需要收縮 | 對 μ_g 套同一 credibility 公式 | 資料給出 **ω ≈ 0.999**（群間 Var(μ)=0.7674 vs 估計誤差 0.00059）→ 無收縮空間 |

**結論**：這些面板的樣本量下，品項自身資料已主導先驗（global pool 的 λ 中位數 0.75–0.90），
partition 沒有施力空間。

**附帶處置**：`--hb-group-shrink-strength 3000` 曾給出 OR MAE 5.5220（勝過 global），
但 (iii) 證明資料驅動權重是 ω≈0.999 而該設定是 ω≈0.11–0.29 —— 那是**對測試集搜尋得到的
過擬合產物，已退役、不入文**。

---

## 3. 懸而未決的問題（需要作者決定，非我能代決）

| # | 問題 | 為何卡住 | 建議 |
|---|---|---|---|
| Q-A | **標題還叫 "Learning to Pool and to Forget" 嗎？** | Round 7 之後，"Pool" 那一半的增益是零。保留可辯（pooling 確實是學的，而「結構選擇是二階」本身是結果），但審稿人會問 | 傾向**保留**，但摘要與貢獻 (3) 必須先講 forgetting、再把 partition 的二階性當作**發現**陳述而非附註。若你偏好保守，改為 "Learning What to Forget: ..." 並把 pooling 降為平面的一個軸 |
| Q-B | Prop 5（雙邊 credibility 界）要不要做？ | 估 1–2 天。做了，Round 7 從負面結果變成理論貢獻；不做，就只是實驗附錄 | **建議做**。這是目前 ROI 最高的一項 |
| Q-C | Round 7 內容放正文還是附錄？ | 正文已估 9–10 頁、超 AISTATS 8 頁 | 正文放 1 段結論 + 1 個小表；機制與三次證偽進附錄 |
| Q-D | AISTATS 2027 CFP 未出 | 頁數上限、template 版本、是否兩階段、checklist 要求 | 等 CFP；搬遷候選已在 `SUBMISSION_CHECKLIST.md` 列好 |
| Q-E | 匿名 repo 未建 | F4 的 Reproducibility Statement 連結是空的 | 投稿前辦（anonymous.4open.science） |
| Q-F | TSB-tuned 附錄 | 你說自行調參補附錄（原 Q6，已暫緩） | 若要進主表，給我參數或預測檔，我重跑對齊協定 |
| Q-G | **整個 v2 工作零 commit** | `paper_v2/`、`docs/`、`tests/`、`scripts/analysis/`、`src/models/mixture_pooling.py` 全為 untracked；`tsb_hb.py`、`run_point.py`、`run_prob.py` modified 未提交 | 我可以立刻分批 commit，等你一句話 |

已裁決、不再是問題：Q1 ablation 處置（接受）、Q2 M5 抽樣描述（屬實）、Q3 DRP 採 (a)、
Q5 GP-Tweedie 引用+附錄、Q6 TSB-tuned 暫緩、Q4 選 (c) 且 (a) 已判定不成立。

---

## 4. 還需要補跑的實驗

### 建議做（有明確論文用途）
| 代號 | 實驗 | 成本 | 用途 |
|---|---|---|---|
| **E1** | **cold-start scaling law**：初始窗截至 14/28/56/全長，量 pooling 增益 vs 有效樣本數 | 低（~1 小時機器時間） | credibility 公式 λ=n/(n+k) 預測增益 ∝ 1/n。實測吻合則「總體無用」變成**被自己理論預測到的定量規律**，同時支撐 Prop 4 與（若做）Prop 5 |
| **E2** | 五面板的 split/credibility ablation 列（目前只有 OR） | 中（~2 小時） | 支撐 Round 7 的全稱陳述；單一面板撐不起「三次證偽」 |

### 選做（有上升空間但不確定）
| 代號 | 實驗 | 成本 | 判斷 |
|---|---|---|---|
| **E3** | soft-responsibility 混合 predictive（不用 hard argmax，θ_i 後驗取 Σ_j π_ij p(θ｜·,j)） | 中高 | **剩下唯一可能真的推動機率指標的路線**：它改變 predictive 的**形狀**（多峰、厚尾），而非只換先驗，正對 Carparts/M5 的 q90 弱點。已列在 Limitations |
| **E4** | per-component 折扣率 w_j | 中 | 讓兩軸交互而非競爭；成則模型貢獻，敗則擴充負面結果 |
| **E5** | G1 庫存模擬（newsvendor / base-stock）、G2 lead-time demand 分位數 | 中 | 原 todolist 的 P2 加分項；把「機率指標更好」翻譯成成本節省 |

### 已決定不做
- B1 GP-Tweedie 重跑 → 改引用 + 附錄說明（Q5 裁決）
- B2 外部階層式 baseline → Limitations 明確聲明（無公開實作可用）
- B3 TSB-tuned → 作者自行處理
- DRP 重跑 → 採 (a)，維持 rebuttal 數字 + caveat + Limitations

---

## 5. 優先順序待辦

### P0 — 一致性與保全（2026-07-29 執行）
- [x] **T1** README 從 v1 論文描述更新為 REMIX（原文還在講 "Taxonomy-Conditioned..."），
      補上 `select`／`--hb-label-split`／`--hb-hyper-shrink` 說明與文件地圖
- [x] **T2** `docs/aistats2027_research_notes.md` 加狀態橫幅（舊工作標題、
      「learned pooling 帶來增益」的暗示已被 Round 7 推翻）
- [x] **T3** `SUBMISSION_CHECKLIST.md` 重寫（B5/B6 早已完成卻仍標未完；
      加入「已決定不做」區塊避免重複追問）
- [x] **T4** Round 6/7 的診斷輸出登錄進 `outputs/aistats2027/MANIFEST.md`，
      含 `scratch_mix_shrink*` 的**退役標記**（R2 可追溯性）
- [x] **T5** 原始稽核清單 `REMIX_revision_todolist.md` 由 Downloads 複製進 repo
- [x] **T6** 建立本檔（STATUS.md）作為單一權威狀態表；
      `QUESTIONS.md`／`CHANGELOG_revision.md` 加上分工指標
- [x] **T6b** ⚠️ **`docs/theory_v2.md` 帶著 D1 已修掉的錯誤常數**（`b/4`、`b*≍(2δ)^{2/3}`），
      與 `main.tex` 直接矛盾——已改為 `b/8` 極限與 `b*=(16δ²)^{1/3}=2^{4/3}δ^{2/3}`，
      並加註以 `tests/test_theory_symbolic.py` 為最終權威
- [x] **T6c** `PAPER_REPRODUCTION.md` 加範圍橫幅（它描述的是 v1 投稿，非當前論文）
- [x] **T6d** 回歸驗證：`tests/test_theory_symbolic.py` 與 `tests/test_determinism.py`
      在 Round 6/7 的程式變動後**皆通過**
- [x] **T7** git commit 完成（ecef8c4 程式／af8685f 測試／5fa694d 論文與文件；後續 Round 8 另行提交）

### P1 — 論文內容（需先解 Q-A / Q-B / Q-C）
- [x] **T8** 摘要 + 貢獻三點改寫完成（Q-A 裁決：**保留標題**）
- [x] **T9** Prop 5（雙邊 credibility）+ Cor 5.1（解析度上限）+ 附錄證明，5 項符號/模擬驗證通過（Q-B 裁決：**做**）
- [x] **T10** §4.3 加「flatness is a resolution limit」段 + 附錄新節 `app:resolution`（2 表 4 段）
- [x] **T11** E1 cold-start scaling law（由既有 slice 資料算出，四面板皆呈幅度隨樣本數衰減）— 尚未畫圖，目前以數字入文
- [x] **T12** E2 四面板修復格子（`tab:repairs`）：RAF −7.1%、Carparts +4.4% 變差，無面板超越 single pool

### P2 — 上升空間
- [ ] **T13** E3 soft-responsibility predictive（剩餘路線中上升空間最大）
- [ ] **T14** E4 per-component w_j
- [ ] **T15** E5 庫存模擬 / lead-time demand

### P3 — 投稿機械作業（等 CFP）
- [ ] **T16** 換官方 template、重測頁數、依 `SUBMISSION_CHECKLIST.md` 搬遷表格
- [ ] **T17** 匿名 repo + 回填 Reproducibility Statement 連結
- [ ] **T18** 移除 `\author`、`ANONYMITY_CHECK.md` 復查、pdflatex ×2

---

## 6. 額外的可發表點（本輪思考中浮現，記錄備用）

以下是超出目前論文範圍、但我認為本身有發表價值的觀察。標記為「入文」者建議寫進本篇；
標記為「可獨立」者可考慮另開一篇或作為 follow-up。

### P-1【入文，並可獨立】折扣會破壞階層模型中變異成分的可識別性
一個一般性現象：任何把**時間近性加權**與**階層／EB 收縮**結合的方法都會遇到。
折扣壓縮有效樣本數 n_eff，使抽樣變異 σ²/n_eff 膨脹；當它超過群內觀測離散度，
變異成分的動差／REML 估計為負而被截斷至 0，credibility 權重隨之歸零——
**模型「正確地」推論出群內品項完全同質，於是丟棄每個品項自己的歷史。**
量化（OR, mixture）：塌陷群數隨 w=1.0/0.98/0.90 為 1/5 → 3/5 → 4/5；
w=0.90 時 37.7% 的序列 λ 精確為 0。
影響範圍不只本篇：DLM 的 discount factor 傳統、global-local 神經預測器的共享先驗，
只要同時做 recency weighting 與 pooling 都適用。我們也給了修法
（Bühlmann credibility 收縮變異成分，權重由資料決定、無可調參數）。

### P-2【入文，接 P-1】λ 分布作為「這個面板值不值得做 pooling」的事前判準
credibility 權重 λ_i = n_i/(n_i + σ²/τ²) **不需擬合任何預測器就能從資料算出**。
本輪結果顯示：pooling 結構只在 λ 明顯小於 1 的地方有施力空間。
這把「我該不該建階層模型」從試錯變成一行診斷。實務價值高，且我沒見過有人這樣陳述。

### P-3【入文，T9】credibility 權重的雙邊界
見 §1 貢獻 (2)。Prop 2 只證了 λ < 1（先驗不被洗掉）；需要對稱的 λ > 0
（品項資料不被洗掉），後者需要 τ² 有下界——正是 P-1 的正則化提供的。
**「折扣能學到多少結構有上限，且該上限可寫成閉式」**是一個新的理論陳述。

### P-4【可獨立】間歇需求的指標病理
全零預測在 5 個面板中的 3 個取得最低 MAE、3 個取得最佳 MASE。
這不是我們模型的問題，是**測量工具的問題**，而 MAE/MASE 是間歇需求文獻的標準指標。
目前作為本篇的一個小節，但它足以支撐一篇獨立的 benchmark／lessons 論文：
「以 MAE/MASE 排名的間歇需求文獻結論不可靠，該用什麼取代」。

### P-5【可獨立，較遠】兩軸交互的參數化
目前 w 是全資料集一個純量，partition 只換先驗，兩軸是**加性、互相競爭**的。
per-component w_j（E4）讓 partition 變成「決定該忘掉誰」的機制。
若成立，這是一個乾淨的模型貢獻，而且正好把標題的兩半縫成一件事。
