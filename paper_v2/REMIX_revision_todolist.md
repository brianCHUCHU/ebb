# REMIX — AISTATS 2027 投稿改稿 TODO List

> 對象：執行端（在論文 repo 內執行）
> 目標稿件：`main.tex`（"Learning to Pool and to Forget"）
> 本清單基於對現有 `main.tex` 的逐項稽核，所有引用的數字都可在 tex 中直接驗證。

---

## 0. 作業守則（開始任何任務前必讀）

這些規則優先於底下所有任務。違反其中任何一條，改稿的價值會是負的。

- **R1｜絕不為了讓文字自圓其說而改動表格數字。** 目前稿中有多處「正文宣稱 ≠ 表格內容」。修法一律是**改正文**，或**重跑實驗後同時更新兩者**。任何情況下都不得反向操作。
- **R2｜任何論文中出現的數字，都必須能追溯到一個 output 檔。** 見任務 `A0`。若某個數字在 repo 裡找不到來源，標記為 `UNVERIFIED` 並回報，**不要猜、不要重算一個近似值填進去**。
- **R3｜不確定時停下來問。** 稿中有些不一致可能是筆誤，也可能反映兩次實驗設定不同（例如 fit window 不同）。這兩種情況的修法完全相反。無法從 code / output 判定時，列入 `QUESTIONS.md` 交回作者裁決。
- **R4｜不得新增未實際執行的實驗結果。** 若某個 baseline 跑不動（例如 DRP 的 MXNet 依賴），照實記錄為 blocked，不要沿用舊數字充數。
- **R5｜每完成一個任務，更新 `CHANGELOG_revision.md`**：任務 ID、改了哪些檔案、哪些數字變動（舊值 → 新值）、來源 output 檔路徑。
- **R6｜雙盲。** AISTATS 為 double-blind。任何新增內容不得包含作者姓名、機構、非匿名 repo 連結、內部版本號、或「我們先前的工作」式表述。
- **R7｜實驗與寫作分離提交。** 先完成 A 組（稽核）與 F 組（編譯/合規），再進 B/C/D/E 組。A 組沒做完就動筆改寫敘述，等於在錯誤的數字上疊敘述。

---

## A 組｜P0：數字一致性稽核（**最優先，會直接影響可信度**）

### `A0` 建立單一數字真相來源
- [ ] 掃描 `outputs/aistats2027/`（或實際輸出目錄），建立 `NUMBERS.md`：把 `main.tex` 中每一個出現在表格、正文、摘要、圖說的數值，對應到 `檔案路徑 : 欄位 : 列` 。
- [ ] 對每個數字標記 `VERIFIED` / `MISMATCH` / `NOT_FOUND`。
- [ ] 額外欄位：該數字是用哪個 fit window 產生的（full training window vs. 80% inner-train），這是 `A3` 的關鍵。
- **驗收**：`NUMBERS.md` 涵蓋率 100%，且 `NOT_FOUND` 清單已回報。

### `A1` 修正 §4.2 的兩處錯誤宣稱（**單一最緊急項目**）
現行文字：
> "REMIX is best or second-best on eight of the ten columns... while no baseline is best on more than two columns, and each panel crowns a different classical leader."

- [ ] **錯誤 1**：實際數 Table `tab:point` 的 REMIX 列，`\best`/`\secondbest` 共 **7 個**，不是 8。
  （OR: RMSSE best、MAE 2nd；M5: 兩欄 best；Auto: MAE best、RMSSE 2nd；RAF: RMSSE 2nd；**Carparts 兩欄皆未進前二、RAF MAE 未進前二**。）
- [ ] **錯誤 2**：TSB 在同表中被 `\best` **三次**（OR MAE `5.574`、Carparts MAE `0.540`、RAF MAE `2.166`）。因此「no baseline is best on more than two columns」為假，「each panel crowns a different classical leader」亦為假。
- [ ] 參考正確計數（請執行端自行重數一次確認）：best 次數 = REMIX 4、TSB 3、SBA 1、ADIDA 1、AutoTheta 1，合計 10。
- [ ] 重寫該句。建議方向：不要打「橫掃欄位數」這種容易被反駁的牌，改打「唯一在五個面板上都不落入尾段、且機率性指標全勝」的一致性論述。
- **驗收**：新句子中每一個計數都能由 `tab:point` 逐格數出來。

### `A2` 修正 §4.4 walk-forward 的自相矛盾
現行文字：
> "Forgetting improves every metric over the stationary configuration however it is configured: ... improves MAE 5.586→5.474 and RMSSE 4.725→4.761"

- [ ] `4.761 > 4.725`，RMSSE 是**變差**的；腳註自己也承認了。刪除 "every metric... however it is configured"。
- [ ] 改寫為：MAE 改善、RMSSE 有小幅代價，並把腳註的解釋（per-block occurrence discounting 換取反應速度）併入正文，不要藏在腳註裡。
- **驗收**：正文與腳註不再互相打臉。

### `A3` 釐清 Table 1 / Table 4 / §4.3 三方數字不一致
- [ ] `tab:point` 的 REMIX（Online Retail）= MAE **5.664** / RMSSE **4.788**。
- [ ] `tab:ablation` 的 `learned + auto`（即 selector 選中的 mixture + 自動 w）= MAE **5.692** / RMSSE **4.790**。
- [ ] §4.3 又稱三種 validation ratio 的結果落在 MAE **5.53–5.66**、RMSSE **4.788–4.793** 的窄帶內 —— 而 5.692 落在帶外。
- [ ] 判定原因並二選一：
  - (a) 若差異來自 **fit window 不同**（Table 1 在完整訓練窗 refit，ablation 在 80% 內訓練窗），則在 `tab:ablation` caption 明確寫出，並在 §4.3 的區間敘述中註明比較基準。
  - (b) 若無法解釋，重跑 ablation 使其與 Table 1 同協定。
- **驗收**：讀者能從 caption 判斷兩表為何不同，且 §4.3 的區間宣稱與被引用的數字一致。

### `A4` 並列值標記規則
- [ ] `tab:point` OR-RMSSE 欄：ADIDA `4.797` 與 IMAPA `4.797` 同值，但只有 ADIDA 標 `\secondbest`。
- [ ] `tab:point` RAF-RMSSE 欄：IMAPA `1.626`、AutoARIMA `1.626`、REMIX `1.626` 三者同值，但只有 REMIX 標 `\secondbest`。
- [ ] 修法二選一：多印一位小數把並列拆開（優先），或在 caption 明訂並列處理規則並一致套用。
- [ ] 對**所有**表格重跑一次並列檢查（`tab:prob`、`tab:walkforward` 亦須檢查）。
- **驗收**：不存在「同值但標記不同」的格子。

### `A5` 覆蓋率與「11% sharper」的基準統一
- [ ] §4.2 寫 "Coverage@80 = 0.90, AIW 10.3"，`tab:coverage` 寫 `0.896` / `10.34`。統一（建議正文直接引表格值，不要另外四捨五入）。
- [ ] 摘要與 §4.2 的 **"11% sharper intervals"** 實際是 `0.837 @ AIW 8.96`（calibrated forgetting）vs `0.845 @ AIW 10.03`（calibrated stationary），約 10.7%。
- [ ] 風險：目前寫法容易被讀成「相對於 `tab:coverage` 的 raw 10.34」。在正文中把比較基準寫死：**兩者皆為啟用 calibration layer 後的配置**。
- [ ] 順帶檢查：`tab:coverage` 的 RAF coverage `0.922` 對 nominal `0.80` 偏離不小，Limitations 應點名（目前只泛稱 "mild over-coverage"）。
- **驗收**：摘要、§4.2、`tab:coverage`、附錄 B 四處的覆蓋率敘述互相一致且基準明確。

### `A6` 顯著性檢定數字對齊
- [ ] `tab:significance` 中 vs. AutoARIMA 的 `$p_t$` 為 `<10^{-15}`，附錄 C.2 為 `<10^{-16}`。統一為 output 檔實際值。
- [ ] 檢查 `tab:significance` 全表與附錄 C.2 逐項對齊（mean diff、$p_t$、$p_W$）。
- [ ] **敘述謹慎化**：vs. TSB 的 `$p_t = 0.079$` 為**不顯著**，但正文 §4.3 寫 "the systematic MAE deficit against TSB is eliminated"。需改為更精確的表述（Wilcoxon 顯著、t-test 不顯著，代表改善集中在中位數而非均值；這其實正好支持你「TSB loss tail 更重」的論點，應該正面利用而非模糊帶過）。
- **驗收**：無 p 值不一致；TSB 段落的統計陳述經得起 statistician reviewer 檢視。

### `A7` 摘要與正文的宣稱逐條核對
- [ ] `19,000+ evaluated series`：3649+5000+3000+2509+5000 = 19,158 ✓（確認即可）。
- [ ] `best mean scaled pinball loss on every dataset` ✓（`tab:prob` 五個 Mean 欄皆 best，確認即可）。
- [ ] `sub-millisecond per-item inference` vs §4.6 的「base fit-predict 0.3 ms/series、**full REMIX selection 6 ms/series**」。摘要的說法對應的是 base 而非完整 pipeline。需改為誠實表述（例如 "sub-millisecond base inference; single-digit millisecond end-to-end including selection"）。
- [ ] `repairs the point-forecast metrics that the stationary hierarchical model loses` —— 這句需與 `C1` 的結論同步（見下）。
- **驗收**：摘要中每個量化宣稱都有表格支撐，且無 over-claim。

---

## B 組｜P1：必補的實驗（審稿人一定會問）

### `B1` 補上 GP–Tweedie 對照（**最高風險缺口**）
- [ ] 你使用 `damato2025` 的 Auto / Carparts / RAF 資料集與其 last-$h$ holdout，卻沒有比較他們的方法。這在審稿時只會有一種解讀。
- [ ] 最低限度：在同一 holdout 下引用其論文原始數字（需確認 metric 定義完全相同才可直接引用，否則必須重跑）。
- [ ] 較佳：實際重跑其開源實作，納入 `tab:point` 與 `tab:prob`。
- [ ] 若 metric scaling 不可對齊，於附錄新增一節說明並提供可對齊的子集比較。
- **驗收**：三個 monthly 面板的兩張主表都出現 GP–Tweedie 列。

### `B2` 補上至少一個外部階層式 baseline
- [ ] §2 把 `chapados2014` / `seeger2016` / `pitkin2024` 列為最接近的競爭者，但實驗中**一個都沒比**；唯一的階層式對照是 REMIX 自己的 $w{=}1$ 版本。
- [ ] 這使「pool but do not forget」的批評缺乏實證支撐。至少納入一個：
  - 優先：`pitkin2024` 的 hierarchical hurdle（有公開實作最佳）；
  - 次選：DeepAR 已在附錄，但那是 neural 而非 hierarchical Bayes，不能替代；
  - 備案：自行實作一個 shared-prior hierarchical Poisson/NB baseline，於附錄說明實作細節。
- [ ] 若全部不可得，必須在 Limitations 明寫「未與外部階層式方法直接比較」，不要靜默略過。
- **驗收**：主表出現外部階層式列，或 Limitations 有明確聲明。

### `B3` 公平調參：TSB 的 smoothing constant 必須同樣被調
- [ ] 目前 baseline TSB 用固定 $(\alpha_d,\alpha_p)=(0.5,0.45)$，而 REMIX 的 $w$ 是在 validation split 上選出來的。
- [ ] 依 Prop 1 的對應關係 $\alpha_p = 1-w$，REMIX 選中的 $w \in [0.90, 0.997]$ 對應 $\alpha_p \in [0.003, 0.10]$ —— 與 baseline 的 `0.45` 差了一個數量級。
- [ ] TSB 又正好是 `tab:point` 中 MAE 最強的 baseline（三個面板奪冠）。**不調 TSB 而調自己，是本稿最容易被攻擊的公平性問題。**
- [ ] 動作：在**同一個 internal validation split** 上對 TSB 的 $(\alpha_d, \alpha_p)$ 做 grid search，新增 `TSB-tuned` 列。Croston / SBA 同理。
- [ ] 若 tuned TSB 追上或超越 REMIX 的點預測，**照實報告**，並把論文重心明確移到機率性指標（那本來就是你更強的地方，且 `kolassa2016` 已替你背書）。
- **驗收**：主表含 tuned 版古典 baseline；調參協定在附錄 B 說明。

### `B4` 補 trivial baseline
- [ ] 加入 **全零預測**（constant-zero）與 **naive / seasonal-naive**。稀疏序列上 constant-zero 的 MAE 常常出奇地強，審稿人很愛問這題；主動放上去反而顯得自信。
- [ ] 至少報 MAE / RMSSE / SPL。
- **驗收**：`tab:point` 與 `tab:prob` 含 trivial baseline 列。

### `B5` M5 抽樣的正當性
- [ ] 目前只用 seed-controlled 5,000 條，而 M5 完整為 30,490 條。M5 恰好是 REMIX 表現最好的面板 —— 審稿人會懷疑抽樣選擇性。
- [ ] 優先：跑完整 M5 bottom-level。你的 pipeline 是 $O(N)$ 且宣稱 6 ms/series，30,490 條約 3 分鐘，**技術上沒有不跑的理由**。
- [ ] 若確有障礙（例如 baseline 的 AutoARIMA 成本），至少：(a) 用多個 seed 重複抽樣並報變異，(b) 在附錄說明抽樣程序與為何不用全量。
- **驗收**：全量結果，或多 seed 穩健性證據 + 明確說明。

### `B6` 跨資料集的多重比較檢定
- [ ] 目前只有單一資料集（Online Retail）的 per-series paired test。預測學界的標準做法是跨資料集的 **Friedman + post-hoc Nemenyi**，並附 critical difference diagram。
- [ ] 另外加 **Diebold–Mariano** 檢定 —— 預測社群對 DM 的接受度高於 Wilcoxon。
- [ ] 產出一張 CD diagram 圖（可取代或補充現有某張圖）。
- **驗收**：新增 §4.5 子節 + CD diagram 圖檔。

### `B7` 補 MASE
- [ ] 現行點預測指標為 MAE / RMSSE。間歇需求文獻的標準指標是 **MASE**（且對 sparse series 比 MAE 更可比）。加一欄成本極低。
- **驗收**：`tab:point` 或附錄含 MASE。

### `B8` 實驗環境規格
- [ ] §4.6 的效率宣稱（0.3 ms / 6 ms / 19 ms per series）沒有硬體規格。補上 CPU 型號、核心數、記憶體、是否單執行緒、Python/套件版本。
- [ ] 釐清 0.3 ms 是否**包含** `B=20` 的 bootstrap（附錄 B 說 released configuration 有做 bootstrap）。若不含，效率表需拆成兩列。
- **驗收**：附錄 B 有完整環境表；bootstrap 成本歸屬明確。

### `B9` 決定論的可驗證性
- [ ] 稿中宣稱 "repeated runs are bit-identical"。加一個 CI 測試：同一設定跑兩次，assert 所有 output CSV 逐位元相同。
- **驗收**：`tests/test_determinism.py` 存在且通過；論文可引用該測試。

---

## C 組｜P1：Ablation 重構與核心論證修補（**最可能導致 reject 的結構問題**）

### `C1` 處理「mixture 總是被選中，但在 ablation 中並不占優」的張力
這是全稿最嚴重的論證問題，請優先處理。

現況：§4.3 的 finding (i) 宣稱 "The learned partition always wins the selection"，但 `tab:ablation`（Online Retail）顯示：

| 配置 | MAE | RMSE | RMSSE |
|---|---|---|---|
| global + auto | **5.533** | 17.839 | 4.790 |
| taxonomy + auto | 5.562 | 17.842 | 4.790 |
| **learned + auto（= selector 選中）** | **5.692** | 17.877 | 4.790 |
| global + off | 5.762 | **17.689** | **4.785** |

也就是說：**selector 選中的配置在該表的三個指標上全都不是最好的**，MAE 甚至輸給最簡單的 global pool，RMSSE 輸給完全不 forgetting 的 stationary corner。目前僅以 "its value concentrates in cold-start slices" 一句帶過，等於承認 selection criterion 與 out-of-sample 表現脫鉤 —— 而「從資料學習結構」正是標題的一半。

- [ ] **`C1a`（必做）**：`tab:ablation` 加上 **SPL（mean 與 q90）欄位**。selector 用的是 pinball，ablation 卻只報 MAE/RMSE/RMSSE，這是不對等比較。極可能 mixture 在 SPL 上確實勝出 —— 但目前讀者拿不到這個證據。
- [ ] **`C1b`**：把 ablation 擴展到**全部五個資料集**（目前只有 Online Retail），放主文或附錄。單一資料集的 ablation 支撐不起「always wins」這種全稱宣稱。
- [ ] **`C1c`**：把 §C.3 的 cold-start / regime slice 結果**提到主文**並量化。若 mixture 的價值真的集中在稀疏切片，那應該是一張主文圖表，而不是附錄一句話。
- [ ] **`C1d`（依 C1a 結果二選一）**：
  - 若加了 SPL 後 mixture 勝出 → finding (i) 保留，但明確寫出「learned partition 的增益在 pinball 上，點指標上與 global pool 相當」。
  - 若仍未勝出 → **誠實改寫 finding (i)**：selector 對 partition 的選擇「無害但非決定性」，真正的增益來自 forgetting 軸。這其實是更乾淨、更可信的故事，且與 `tab:ablation`「forgetting 在每種 pooling 下都改善 MAE」完全一致。同時需下修摘要與 §1 Contributions 中對 learned pooling 的措辭。
- [ ] **`C1e`**：把 finding (iii)「the plane is flat near its optimum」**提前**到 finding (i) 之前。先建立「這個平面很平、selector 的工作是避開壞角落」的框架，再談 partition 選擇，敘事會穩健得多。
- **驗收**：`tab:ablation` 含 SPL；五資料集版本存在；finding (i) 的措辭與證據一致，不再有「宣稱勝出但表格顯示落後」的落差。

### `C2` Ablation 的 mixture $K$ 敏感度
- [ ] 目前 $K$ 由 BIC 選（$K\in\{4,5,6\}$）。加一組 $K$ 固定為 $\{2,3,\dots,6\}$ 的敏感度結果，證明 BIC 選擇不是關鍵超參數。
- **驗收**：附錄含 $K$ 敏感度表。

### `C3` Online Retail 評估協定的正當性
- [ ] 現行協定：「first third initializes, rest evaluates」—— 等於用固定預測跨越約 2/3 年的 horizon。這種協定在結構上偏好「穩定強度估計」的方法，可能被質疑對 REMIX 有利。
- [ ] 動作：(a) 在 §4.1 明確說明並正當化此協定（例如對應真實補貨週期），且 (b) 已有 walk-forward 作為補充 —— 需在 §4.1 就前向指出，而非等到 §4.4。
- [ ] 加 horizon-wise 分解圖（誤差 vs. horizon），顯示優勢不是只來自長 horizon 的平均效應。
- **驗收**：協定有明說的理由 + horizon 分解證據。

---

## D 組｜P1：理論補強（對 AISTATS 而言目前偏薄）

現況診斷：Prop 1 是代數恆等式，Prop 2 是一行不等式，Prop 3 是標準 bias–variance 平衡。數學本身正確，但份量不足以撐起 AISTATS 的理論賣點，且**三個命題全在 forgetting 軸上 —— pooling 軸（標題的另一半）沒有任何理論**。

### `D1` 修正 Corollary 1 的常數不一致（**數學錯誤，必修**）
- [ ] Prop 3 的變異數界為 $\tfrac14(1-w)(1+w^T)/[(1+w)(1-w^T)]$。令 $b=1-w$、$T\to\infty$，其極限為 $\tfrac{1-w}{4(1+w)} \to b/8$。
- [ ] 但附錄 A.3 的證明寫 "variance $\asymp b/4$"，而其推出的 $b^{*}\asymp(2\delta)^{2/3}$ 實際對應的是 **$b/2$**。三個常數互不相容。
  （驗算：minimize $\delta^2/b^2 + cb$ 得 $b^*=(2\delta^2/c)^{1/3}$。$c=1/2 \Rightarrow b^*=(4\delta^2)^{1/3}=(2\delta)^{2/3}$；$c=1/8 \Rightarrow b^*=(16\delta^2)^{1/3}$。）
- [ ] $O(\delta^{2/3})$ 的**速率不受影響**。最簡修法：Corollary 改寫為 $1-w^{*}\asymp \delta^{2/3}$，把明確常數在證明中算一次並保持一致。
- [ ] 順帶：Prop 1 與 Prop 2 的證明我已核對無誤，但仍請用 `sympy` 對三個命題做符號驗證，並把驗證腳本放進 `tests/`。
- **驗收**：命題、系理、附錄證明三處常數一致；符號驗證腳本通過。

### `D2` 新增：selector 的 oracle inequality（**最高價值的理論補強**）
- [ ] 你在有限候選集（8 個 $w$ × 3 種 partition = **24 個候選**）上以 chronological hold-out 做選擇 —— 這正是 finite-class validation / model selection oracle inequality 的教科書設定。
- [ ] 寫下大約半頁的結果：選中配置的風險以高機率不超過 oracle 加上 $O\big(\sqrt{\log 24 / n_{\text{val}}}\big)$ 量級的項（需先確立 pinball loss 在你的預測分布下有界或 sub-Gaussian，這點要說清楚）。
- [ ] 這一擊同時做到三件事：(a) 直接支撐論文核心主張「這兩個決策應該從資料學」；(b) 補上 pooling 軸的理論空缺；(c) 解釋 §4.3 的 "plane is flat" 經驗觀察。**目前這塊是空的，而它本該是全稿最強的論證。**
- [ ] 記得處理時間相依性（hold-out 是時間序列的尾段，非 i.i.d.）—— 可用 blocking / mixing 假設，或誠實把它列為假設。
- **驗收**：新增 Proposition 4 + 證明 + 與 §4.3 經驗結果的連結段落。

### `D3` 補 pooling 軸的理論陳述（次要，若 D2 已做可降級）
- [ ] 至少給一個 James–Stein 式或 Bühlmann credibility 式的陳述：在何種條件下 pooled estimator 的風險優於 per-item MLE。這是經典結果，引述並套用到你的設定即可，不需原創。

### `D4` 把 Complexity 段落搬離 Theory 節
- [ ] §3.5 底下的 `\paragraph{Complexity}` 內容與理論無關，獨立為 §3.6。

---

## E 組｜P1：文獻（以下幾條不引很危險）

### `E1` 必補引用
- [ ] **West & Harrison, _Bayesian Forecasting and Dynamic Models_** — discount factors / power-discounted likelihood。對充分統計量做指數折扣在 DLM 傳統中 1970s 就有。不引，審稿人會直接說 forgetting operator 是舊酒新瓶。**引了並說明差異**（你把折扣套在 **EB 超參數擬合**上，而非僅狀態演化）反而變成賣點。
- [ ] **Ibrahim & Chen (2000), power prior** — §3.2 正文直接用了 "power-prior interpretation" 卻沒引來源。
- [ ] **Bühlmann (1967) / Bühlmann–Straub credibility** — 你的 $\lambda_i = n_i/(n_i+\phi_g)$ 就是精算的 credibility factor，一字不差。保險文獻做這件事六十年了。承認它，同時借用其現成直覺與理論工具（也直接支援 `D3`）。
- [ ] **Snyder, Ord & Beaumont (2012)** — 間歇需求 count-model 的標準參考。
- [ ] **Montero-Manso & Hyndman (2021)**, "Principles and algorithms for forecasting groups of time series: locality and globality" — 直接對應你的 pooling 軸，是 global vs. local 的核心參考。

### `E2` 建議補充
- [ ] Petropoulos & Kourentzes 關於 intermittent demand 分類與 aggregation 的後續工作。
- [ ] Gneiting & Raftery (2007) — proper scoring rules，支撐你以 pinball 為選擇準則的正當性。
- [ ] Conformal 時間序列的較新工作（如 adaptive conformal inference），因為你的 CP baseline 用的是 split conformal 的靜態版本，reviewer 可能指出這對 conformal 不夠公平。

### `E3` Related Work 改寫
- [ ] 新增一段「discounting / power priors in Bayesian forecasting」，明確定位 REMIX 相對於 DLM discount factor 傳統的貢獻。這段目前完全不存在，是最大的文獻缺口。
- [ ] 新增一段 credibility theory 的連結。

---

## F 組｜P0：編譯、格式與雙盲合規

### `F1` 會影響編譯或輸出的問題
- [ ] `\newcommand{\ind}[1]{\mathbb{1}\{#1\}}` — `amssymb` 的 `\mathbb` 只涵蓋大寫拉丁字母，`\mathbb{1}` 通常吐 `Missing character` 或印出空白。改用 `bbm` 的 `\mathbbm{1}`、`dsfont` 的 `\mathds{1}`，或直接 `\mathbf{1}`。**改完務必實際編譯並目視檢查 §3.1 的輸出。**
- [ ] `\bibliographystyle{plainnat}` 與手寫 `thebibliography` 併用是無效指令，刪除。
- [ ] `tab:neural` 的 `\begin{table}[h]` 改為 `[t]` 或 `[htbp]`。
- [ ] `figs/fig1_overview` 缺副檔名，其他圖皆有 `.pdf`。統一。
- [ ] 移除未使用項：`tikz` 與 `arrows.meta`（全文未用）、`\newtheorem{remark}`（未用）。
- **驗收**：`pdflatex` 兩次無 warning（除已知的 overfull box）；PDF 中指示函數符號正確顯示。

### `F2` 雙盲風險掃描
- [ ] §4.5：`"matching the prior-sensitivity findings reported for the stationary model"` — 無引用對象，讀起來像在指自己先前的工作。改為明確引用他人文獻，或刪除該比較。
- [ ] 附錄 C.5：`"The v2 additions change none of the inference complexity"` — 內部版本號洩漏，且暗示存在 v1。刪除或改寫。
- [ ] 附錄 C.5：`"reported verbatim from the identical-protocol comparison"`（DRP 因 MXNet 不可重跑）— 除 anonymity 外，**rigor 上也弱**：報一組現在跑不出來的數字，多數審稿人會要求整段刪除或移入 limitation。建議：能重跑就重跑（考慮改用 PyTorch 實作或 GluonTS 的現代等價模型）；不能就整段移到 Limitations 並明說無法驗證。
- [ ] 附錄 B 提到的 `outputs/aistats2027/` 與 `remix_selection_diagnostics.csv` — 確認路徑字串本身不含作者/機構識別。
- [ ] 全文搜尋：作者名、機構名、致謝、grant 編號、非匿名 URL、`\thanks`。
- **驗收**：`ANONYMITY_CHECK.md` 記錄掃描結果，全部清空。

### `F3` 投稿規格
- [ ] 目前雙欄 10pt + 4 圖 5 表，主文估計 **9–10 頁**。AISTATS 一向是主文 8 頁（不含參考文獻與 supplementary）。
- [ ] **等 AISTATS 2027 CFP 出來後確認**：頁數上限、是否仍為兩階段投稿（初審 + supplementary 補件）、是否要求 checklist、template 版本。
- [ ] 依頁數規劃搬遷：`tab:significance`、`tab:coverage`、`tab:walkforward` 中至少一張可移入附錄；§4.4 walk-forward 可壓縮為一段 + 附錄表。
- [ ] 換用官方 AISTATS style file 後重新排版並重測頁數。
- **驗收**：符合官方頁數；`SUBMISSION_CHECKLIST.md` 建立。

### `F4` 可重現性
- [ ] 論文中目前**沒有 code 連結**。建立匿名 repo（anonymous.4open.science 或類似），在論文中加入連結。
- [ ] 新增 **Reproducibility Statement** 一節（多數 ML 會議要求）：資料取得方式、前處理腳本、seed 政策（你的賣點是無 seed，要講清楚）、硬體、預估執行時間。
- [ ] 附錄 C.1 提到 `MANIFEST.md` 映射表格到來源檔 —— 確認它真的存在且完整（與 `A0` 併做）。
- **驗收**：匿名連結可用；Reproducibility Statement 完成。

---

## G 組｜P2：加分項（時間允許再做）

### `G1` 庫存模擬評估
- [ ] 論文反覆以「inventory decisions live at high quantiles」（`kolassa2016`）作為動機，但從未實際做庫存評估。加一個 newsvendor / base-stock 模擬：在給定 service level 下比較 holding cost + stockout cost。
- [ ] 這會把「機率性指標更好」直接翻譯成「錢更省」，是對 application-track 審稿人最有說服力的一擊。

### `G2` Lead-time demand（聚合 horizon）評估
- [ ] 真實補貨看的是 lead-time 內的累積需求分布，而非單期。加一個 $L$-step aggregated demand 的分位數評估（$L \in \{7, 14, 28\}$）。

### `G3` Limitations 中承諾的延伸至少實作一項
- [ ] 現行 Limitations 列了四項（Gamma size block、soft-responsibility mixture、full posterior、criterion alignment）。其中 **criterion alignment 你已經有數據**（§4.7 提到選 MAE 準則會得到 global pool + $w{=}0.90$，OOS MAE `5.508`，優於 `tab:ablation` 中所有配置）。
- [ ] 這個結果目前被埋在 Limitations 一句話裡 —— **它其實是個正面結果**，應該提升為 §4 的一個小節：「selection criterion 可依部署損失對齊」。
- [ ] 若時間允許，實作 Gamma size block（同一架構下共軛），會直接回應 Carparts / M5 q90 落後的問題。

### `G4` 圖表品質
- [ ] 確認所有圖為向量 PDF、字體嵌入、在灰階列印下可讀（AISTATS 審稿人常印出來看）。
- [ ] Figure 1 目前是全寬 `figure*`，佔版面成本高；在頁數壓力下評估是否縮為單欄。

---

## H 組｜寫作與結構（在 A–D 組數字定案後執行）

- [ ] `H1` **重寫摘要** —— 待 `A1`、`A7`、`C1d` 定案後進行。移除已被證偽的宣稱，把重心明確放在「機率性指標全面領先 + 理論解釋 selector 行為」。
- [ ] `H2` **§4.2 整段重寫**（依 `A1`）。
- [ ] `H3` **§4.3 finding (i)/(iii) 重寫並調序**（依 `C1d`、`C1e`）。
- [ ] `H4` **§4.4 walk-forward 段重寫**（依 `A2`）。
- [ ] `H5` **Limitations 提升為獨立一節**（目前是 §4 的子節，位階不對），並依 `B2`、`F2` 新增缺失的自我聲明。
- [ ] `H6` **Complexity 獨立為 §3.6**（依 `D4`）。
- [ ] `H7` §1 Contributions 三點依最終結果重寫；特別是 (2)、(3) 的量化措辭。
- [ ] `H8` 新增 notation table（附錄），目前符號量已達到需要索引的程度。
- [ ] `H9` 全文 over-claim 掃描：搜尋 `every`、`always`、`all`、`never`、`exactly`、`dominates`，逐一確認證據支撐（`\REMIX{} dominates the entire profile` 這類句子尤其要核）。
- [ ] `H10` 統一術語：`forgetting configuration` / `hierarchical, forgetting` / `discounting` 三種說法在文中混用，指的是同一件事。

---

## 建議執行順序

```
第 1 輪（不動實驗，1 天內可完成）
  A0 → A1 A2 A4 A6 → F1 F2 → D1
  產出：QUESTIONS.md、NUMBERS.md、可編譯且無數字錯誤的稿件

第 2 輪（需重跑，決定論文成敗）
  A3 A5 → C1a C1b C1c → B3 B4 → C1d
  產出：擴充後的 ablation、公平的 baseline、finding (i) 的最終定調

第 3 輪（補齊審稿人必問）
  B1 B2 B5 B6 B7 B8 B9 → C2 C3
  產出：完整 baseline 陣容與統計檢定

第 4 輪（理論與文獻）
  D2 → D3 D4 → E1 E2 E3
  產出：Proposition 4 + Related Work 新段落

第 5 輪（寫作定稿）
  H1–H10 → F3 F4 → G 組（視時間）
```

---

## 若時間只夠做三件事

1. **`A1`** — §4.2 的「eight of ten / no baseline more than two」目前是錯的，而且錯在最顯眼的位置。
2. **`C1`** — 給 `tab:ablation` 加 SPL 欄，處理「mixture 總是被選中卻不占優」的張力（或誠實重寫 finding (i)）。
3. **`B1` + `E1`** — 補 GP–Tweedie 對照，補 West & Harrison / Bühlmann 兩條文獻線。

---

## 交回作者裁決的問題（執行端請填入 `QUESTIONS.md`）

- Table 1 的 REMIX(OR) `5.664` 與 Table 4 的 `learned+auto` `5.692` 差異，是 fit window 不同，還是其中一個是舊版數字？
- M5 只用 5,000 條是算力限制，還是有其他原因？
- DRP 的 MXNet 環境是否值得投入時間修復，或直接改用現代等價實作？
- 是否接受「learned pooling 的增益不在 aggregate 點指標」這個較謙虛的定位（若 `C1a` 的 SPL 結果不支持現行宣稱）？
- 是否有算力預算跑完整 M5 + tuned baselines + GP–Tweedie？
