# 設定詳解

設定檔是一份 JSON，路徑由 `--config` 指定。載入時會嚴格驗證：**未知欄位直接報錯**，避免打錯字卻默默生效。

## 必填欄位

| 欄位 | 說明 |
|---|---|
| `root` | 專案根目錄（相對或絕對路徑） |
| `db` | SQLite 檔名，相對於 `root` |
| `owner_id` | 唯一擁有者的內部 ID；非此 ID 的訊息一律拒絕 |
| `character_id` | 角色 ID，貫穿所有記憶資料 |
| `persona_file` | 人設檔路徑（相對於 `root`）；缺失或空白會啟動失敗 |
| `l0_dir` | L0 分層記憶目錄 |
| `model_url` | OpenAI 相容端點，必須是 loopback 或在允許清單中 |
| `model_id` | 送給端點的模型名稱 |
| `telegram_token_env` | 存放 bot token 的**環境變數名稱** |
| `telegram_user_id_env` | 存放擁有者 Telegram ID 的**環境變數名稱** |

## 選填欄位

| 欄位 | 說明 |
|---|---|
| `addressee_family_key` | 指定「家人」類記憶的 key（例如 `家人/配偶`）；不填則停用該功能 |
| `model_api_key_env` | 端點金鑰的環境變數名稱；非 loopback 端點必填 |
| `model_choices` | 多後端清單，啟用 `/model` 切換 |

### `model_choices`

每一項需要 `label`（顯示名稱，唯一）、`model_url`、`model_id`，非 loopback 端點另需 `model_api_key_env`。第一項為預設。

## 端點允許清單

非 loopback 的端點必須明列在 `bot_config.py` 的 `approved_remote_model_urls()`，否則設定驗證會失敗。這是刻意的：避免設定被改動後把 API key 送去未知網域。

新增端點時，請同時在 `bot_config.py` 加入常數，並在測試中補一筆驗證。

## 憑證

- 設定檔只存**變數名稱**，值一律從環境讀取。
- `.env` 已被 `.gitignore` 排除；不要把金鑰寫進 JSON、程式或文件。
- 啟動時會檢查金鑰是否存在且格式合理，缺漏即以啟動錯誤結束（fail closed）。

## 思考（thinking）

- `THINK_MODE` 未設定 → 對支援的端點帶 `reasoning_budget_tokens=128`
- `THINK_MODE=off` → 明確關閉思考
- `THINK_MODE=50`～`150` → 覆寫預算
- 其他值 → 啟動錯誤（不接受模糊設定）

預設值來自實測：低比特量化的地端模型在完全關閉思考時容易短回或空回；預算過大（如 150 以上）反而更差。

## 執行期狀態

- `model-choice.json`：`/model` 的選擇，放在資料庫同目錄；刪除即回到預設後端
- `*.sqlite3` / `*.db`：對話與記憶資料，已被 `.gitignore` 排除

## 檢查模式

```bash
python bootstrap_test_bot.py --check --config bot.json
```

只驗證設定、路徑、人設與端點，不啟動服務、不連 Telegram、不呼叫模型。
