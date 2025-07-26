## 🔧 使用 uv 進行套件與環境管理

本專案採用 [`uv`](https://github.com/astral-sh/uv) 作為套件與虛擬環境管理工具，替代 pip、venv 以及 pip-tools。

### 📦 安裝 uv

你可以使用以下方式安裝 `uv`：

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
# 或 macOS (使用 Homebrew)
brew install astral-sh/uv/uv
```

確認版本：

```bash
uv --version
```

---

### 🚀 建立並啟用虛擬環境

在專案根目錄中執行：

```bash
uv venv
source .venv/bin/activate        # Linux/macOS
# 或 .venv\Scripts\activate.bat  # Windows
```

---

### 📥 安裝依賴套件

使用 `pyproject.toml` 中的設定安裝：

```bash
uv pip sync
```

---

### ➕ 新增套件

使用 `uv add` 可以自動加入依賴並更新 `pyproject.toml`：

```bash
uv add opencv-python
uv add sentence-transformers
```

新增開發用依賴：

```bash
uv add --dev black pytest
```

---

### ❌ 移除套件

```bash
uv remove package-name
```

---

### 🔄 更新依賴

```bash
uv add some-package@latest
```

---

### 🧪 執行測試（如果有 pytest）

```bash
pytest
```

---

### 📁 `.gitignore` 建議

請將虛擬環境排除在版本控制外：

```
.venv/
__pycache__/
*.pyc
*.log
```

---

### 📚 更多資訊

請參閱官方文件：https://github.com/astral-sh/uv
