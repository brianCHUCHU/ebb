# FAILURES — integrity_2026-08-06

## [task1] 驗收失敗：corrected selection 未複現 v4 稿的結構敘事

- 預期（裁決前提）：OR/Carparts/RAF = taxonomy、Auto = global、M5 = mixture。
- 實際（leakage-safe head-only 標籤、既有評分函式）：
  OR (global, 0.95)、Carparts (global, 0.90)、RAF (mixture, 0.997)、
  Auto (global, 0.99)、M5 (mixture, 0.95)。
- 符合 2/5（M5、Auto 的結構；兩者的 w 皆不符）。
- 三個不符面板的結構邊際全部 <0.1%（OR 0.051%、Carparts 0.082%、
  RAF 0.017%）——在選擇器噪音之內。
- 依指引：停止，未執行任務 2/3，未改動 .tex，等候裁決。
- 佐證洩漏方向本身成立：head-only 後 mixture 全面優勢消失
  （Auto mixture 由 +2.2% 反轉為 −1.7%）。

## [task4] 稿中 (0.54%, 1.4%) 無法溯源

- 全曲面 × 全折扣 × 全定義窮舉（±0.03pp）：1.4% 零命中；0.54% 僅散落
  近似命中，無一與 1.4% 同曲面同折扣共存。
- 手算引用的 0.75320/0.74434 屬 OR legacy 曲面（標註 M5 seed 44 有誤）。
- 詳見 NOTES.md 任務 4 節。
