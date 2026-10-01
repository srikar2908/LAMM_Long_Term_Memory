# Running LAMM in Visual Studio Code (Complete Guide)

This guide provides step-by-step instructions to set up, run, debug, and evaluate the **LAMM (Lightweight Adaptive Memory Management)** project inside **Visual Studio Code** on Windows using the pre-configured virtual environment `lammenv`.

---

## 1. Prerequisites & Environment Overview

- **Operating System:** Windows 10/11
- **Code Editor:** Visual Studio Code (VS Code)
- **Python Version:** Python 3.12 (inside `lammenv`)
- **Virtual Environment Path:** `c:\Users\srika\LAMM\lammenv`
- **Installed Packages:** `fastapi`, `uvicorn`, `streamlit`, `faiss-cpu`, `sentence-transformers`, `pydantic`, `pandas`, `matplotlib`, `pytest`, etc.

---

## 2. Opening the Project in VS Code

1. Open **VS Code**.
2. Click **File > Open Folder...** (or press `Ctrl + K, Ctrl + O`).
3. Select the root folder: `c:\Users\srika\LAMM`.
4. Ensure the workspace trust dialog is accepted if prompted ("Yes, I trust the authors").

---

## 3. Selecting the Python Interpreter (`lammenv`)

VS Code needs to know which Python environment to use for IntelliSense, code completion, debugging, and terminal sessions.

### Method A: Command Palette (Recommended)
1. Press `Ctrl + Shift + P` (or `F1`) to open the Command Palette.
2. Type and select: `Python: Select Interpreter`.
3. Select the interpreter pointing to your project environment:
   ```text
   Python 3.12.x ('.\lammenv': venv) .\lammenv\Scripts\python.exe
   ```
4. If it does not appear in the list, click **Enter interpreter path...** -> **Find...** and browse to:
   ```text
   c:\Users\srika\LAMM\lammenv\Scripts\python.exe
   ```

### Method B: Automatic Configuration
The repository includes `.vscode/settings.json` which automatically sets:
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/lammenv/Scripts/python.exe"
}
```

---

## 4. Recommended VS Code Extensions

For the best development experience, install these extensions from the VS Code Marketplace (`Ctrl + Shift + X`):
- **Python** (`ms-python.python`) — Official Python support
- **Pylance** (`ms-python.vscode-pylance`) — Fast, feature-rich language support
- **Python Debugger** (`ms-python.debugpy`) — Debugging integration

---

## 5. Running from VS Code Integrated Terminal

Open the built-in terminal in VS Code using ``Ctrl + ` `` (backtick) or **Terminal > New Terminal**.

> **Note:** Ensure your terminal uses PowerShell or Command Prompt. The `lammenv` environment will typically auto-activate (showing `(lammenv)` in the prompt). If not, activate it manually:

```powershell
.\lammenv\Scripts\activate
```

### Quick Commands:

| Action | Command | Purpose |
| :--- | :--- | :--- |
| **Run Offline Demo** | `.\lammenv\Scripts\python.exe scripts\run_demo.py` | Runs the 9-turn conversational demo showing all lifecycle decisions. |
| **Run Evaluation Suite** | `.\lammenv\Scripts\python.exe scripts\run_evaluation.py` | Executes 3-way evaluation and generates charts, tables, and reports in `results/`. |
| **Run Pytest Suite** | `.\lammenv\Scripts\python.exe -m pytest` | Executes all 32 unit and integration tests. |
| **Start FastAPI Server** | `.\lammenv\Scripts\uvicorn.exe app.main:app --reload` | Launches backend API at `http://127.0.0.1:8000`. |
| **Start Streamlit UI** | `.\lammenv\Scripts\streamlit.exe run dashboard\streamlit_app.py` | Launches interactive dashboard at `http://localhost:8501`. |
| **Gemini Smoke Test** | `.\lammenv\Scripts\python.exe scripts\run_gemini_smoke.py` | Tests Gemini API connectivity (requires `.env` key). |

---

## 6. One-Click Running & Debugging (F5)

The repository provides pre-configured launch tasks in `.vscode/launch.json`.

1. Click on the **Run and Debug** icon in the Activity Bar on the left (or press `Ctrl + Shift + D`).
2. At the top of the panel, click the dropdown menu next to the green play button.
3. Choose one of the 5 pre-configured targets:

### 1. `LAMM: Run Demo (scripts/run_demo.py)`
- Runs the deterministic 9-turn conversation.
- You can set breakpoints inside `app/memory/lifecycle.py` or `app/memory/manager.py` to inspect scoring and decision making turn-by-turn!

### 2. `LAMM: Run Evaluation (scripts/run_evaluation.py)`
- Executes the fair 3-way benchmark comparison (LAMM vs UNMANAGED_MEMORY vs RECENCY_ONLY).
- Generates all 6 PNG figures in `results/figures/` and markdown report in `results/reports/`.

### 3. `LAMM: FastAPI Server (uvicorn)`
- Starts the FastAPI application with auto-reload.
- Open your browser to: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the interactive Swagger API documentation.

### 4. `LAMM: Streamlit Dashboard`
- Starts the Streamlit dashboard on port `8501`.
- Open [http://localhost:8501](http://localhost:8501) to explore the 4 pages:
  - **Chat Interface:** Live conversational agent with real-time memory extraction and lifecycle inspection.
  - **Memory Inspector:** Browse active, archived, and forgotten records in SQLite.
  - **Analytics:** Visual charts of lifecycle distribution and access metrics.
  - **3-Way Comparison:** Side-by-side benchmark comparison.

### 5. `LAMM: Run Pytest Suite`
- Runs all 32 unit tests with verbose output in the debug console.

---

## 7. Using the VS Code Test Explorer

VS Code has a built-in graphical test runner:
1. Click on the **Testing** icon (the beaker icon) on the left sidebar.
2. Click **Run Tests** (or the play icon next to individual test files).
3. All 9 test modules under `tests/` will be discovered:
   - `test_api.py` (4 tests)
   - `test_baseline.py` (2 tests)
   - `test_deduplication.py` (2 tests)
   - `test_embeddings.py` (3 tests)
   - `test_end_to_end.py` (2 tests)
   - `test_lifecycle.py` (8 tests)
   - `test_retrieval.py` (4 tests)
   - `test_scoring.py` (4 tests)
   - `test_storage.py` (3 tests)
4. You can click on any individual test to run it, debug it with breakpoints, or inspect failures.

---

## 8. Step-by-Step Walkthrough for Presentation / Demo

If presenting the project to a review panel or evaluator:

### Step 1: Show the Clean Test Suite
Open the terminal in VS Code and run:
```powershell
.\lammenv\Scripts\python.exe -m pytest -v
```
*Result:* Shows all 32 tests passing cleanly in offline deterministic mode.

### Step 2: Run the Deterministic Demo
```powershell
.\lammenv\Scripts\python.exe scripts\run_demo.py
```
*Explain:*
- Point out Turn 1-5 (`RETAIN`): Initial user preferences are captured.
- Point out Turn 6 (`UPDATE`): User switches from Python to Java; the controller detects supersession and modifies the existing record instead of adding a duplicate.
- Point out Turn 7 (`ARCHIVE`): Low confidence memory is archived out of active index.
- Point out Turn 8 (`COMPRESS`): Verbose memory is condensed to save token budget.
- Point out Turn 9 (`FORGET`): Ephemeral filler memory is dropped.
- Point out Final Query: When asked *"What programming language am I currently using for my backend project?"*, the agent correctly retrieves Java!

### Step 3: Run the 3-Way Comparative Evaluation
```powershell
.\lammenv\Scripts\python.exe scripts\run_evaluation.py
```
*Explain:*
- Show that all three systems receive identical turn sequences and queries.
- Open `results/reports/evaluation_report.md` in VS Code Markdown Preview (`Ctrl + Shift + V`).
- Open `results/figures/active_memory_count.png` and `results/figures/token_context_consumption.png` in VS Code to show measured reductions.

### Step 4: Launch the Streamlit Research Dashboard
```powershell
.\lammenv\Scripts\streamlit.exe run dashboard\streamlit_app.py
```
- Open `http://localhost:8501`.
- Demonstrate the live chat, inspect memory records in the database, and showcase the comparison charts.

---

## 9. Troubleshooting Common Windows / VS Code Issues

### Issue 1: PowerShell Script Execution Policy Error
*Error:* `File ... activate.ps1 cannot be loaded because running scripts is disabled on this system.`  
*Solution:* Run PowerShell as Administrator and run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Alternatively, directly use the executable paths without activation:
```powershell
.\lammenv\Scripts\python.exe scripts\run_demo.py
```

### Issue 2: Port Already in Use (FastAPI on 8000 or Streamlit on 8501)
*Solution:* Kill any running python processes or use an alternate port:
```powershell
.\lammenv\Scripts\uvicorn.exe app.main:app --port 8001
.\lammenv\Scripts\streamlit.exe run dashboard\streamlit_app.py --server.port 8502
```

### Issue 3: Unicode Character Display in Windows Console
*Note:* The codebase has been standardized with ASCII indicators (`->`, `*`, `[Turn N]`) to ensure seamless execution across all Windows consoles (Command Prompt, PowerShell, Windows Terminal) without cp1252 encoding crashes.

---

*LAMM: Lightweight Adaptive Memory Management for Long-Term LLM Agents — B.Tech Mini Project*
