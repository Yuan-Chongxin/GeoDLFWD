# GeoDLFWD

GUI tool for magnetotelluric (MT) deep-learning forward modeling. Built with PyQt5 and PyTorch. Supports data import, model configuration, training, prediction, and sharing models/data via GitHub (including Git LFS for large files).

**Author:** ycx  
**Creation Time:** 2025

---

## Features

- **Data import:** MT modes (TE / TM / Both), import resistivity and phase data from local files.
- **Param & model config:** Choose network architecture, input/output channels, training hyperparameters.
- **Model training:** Integrates `MT_train.py`; supports resume from checkpoint, early stopping, validation split.
- **Forward prediction:** Load trained models, run forward modeling, and visualize results (canvas supports zoom, pan, right-click to save image).
- **GitHub collaboration:** Upload models/data from the Help menu (with Git LFS for large files); download shared resources from the Collaboration tab.

---

## Requirements

- **Python:** >= 3.9
- See `requirements.txt` for dependencies.

---

## Install & Run

### 1. Clone or download the repo

```bash
cd dl   # or your project root
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For a mirror (e.g. China):

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. Start the application

```bash
python GeoDLFWD.py
```

---

## Dependencies

| Category      | Packages |
|---------------|----------|
| GUI           | PyQt5 |
| Scientific    | numpy, scipy, pandas |
| Visualization | matplotlib |
| Deep learning | torch, torchvision |
| Image         | opencv-python, scikit-image |

See `requirements.txt` for versions.

---

## Project structure

```
dl/
├── GeoDLFWD.py              # Main entry and main window
├── ml_trainer.py            # Training thread and MT_train invocation
├── MT_train.py              # MT training script
├── MT_test.py               # Test / prediction
├── ParamConfig.py           # Training and forward parameters
├── PathConfig.py            # Path configuration
├── LibConfig.py             # Library and environment config
├── func/                    # Data loading, network definitions, etc.
├── nn/                      # Network modules (FCN, ResNet, VGG, etc.)
├── cei/                     # CEI-related modules
├── layers/                  # Layered modules
│   ├── collaboration_layer/   # GitHub adapter, etc.
│   ├── ui_layer/               # Collaboration tab, upload dialog
│   └── ...
├── data/                    # Example data and models
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Usage

- **Data and model paths:** Configure in the Data Import and Param Config tabs; paths can be set via `PathConfig` etc.
- **Training:** Set parameters in Model Training and start; resume from existing checkpoint is supported.
- **Prediction:** In Model Prediction, select model and data, then run; results are shown on the canvas (zoom, pan, right-click to save image only — no system context menu).
- **Upload (Help menu):** Use **Help → Upload model** or **Upload data** to push files to GitHub; for files over 25MB, Git LFS is recommended (see below).
- **Download & collaboration:** In the **Collaboration** tab, set repository and Token, then refresh and download shared models or data.

---

## Help menu: Upload model / Upload data

### Overview

The Help menu provides **Upload model** and **Upload data**, sending trained models and data files to a GitHub repository. Files over 25MB are handled via Git LFS (Large File Storage).

### How to use

1. **Open upload:** Menu **Help (H)** → **Upload model (M)** or **Upload data (D)**.
2. **Configure GitHub:**
   - **Repository:** Owner (default "Yuan-Chongxin"), repo name (default "GeoDLFWD"). Adjust as needed.
   - **GitHub Token:** Personal Access Token with `repo` scope. Use **Connect** to test.
   - **Status:** Shows connection state (not connected / connected / Git not installed).
3. **Select file:** Choose file; supported types include:
   - **Models:** `.pth`, `.pt`, `.h5`, `.ckpt`, `.pkl`
   - **Data:** `.txt`, `.csv`, `.mat`, `.npy`
   File size is shown; if >25MB, Git LFS is suggested.
4. **Upload mode:** Standard (files &lt;25MB) or **Use Git LFS** (recommended for &gt;25MB).
5. **Commit message:** Optional; a default is used if left empty.
6. **Upload:** Click the upload button and check the log for progress and result.

### Git LFS

- **What it is:** Git LFS handles large files. GitHub warns above 25MB and has a 100MB file limit; LFS avoids these issues.
- **Install:**
  - **Windows:** https://git-lfs.github.com/ → install → then run `git lfs install`.
  - **macOS:** `brew install git-lfs` then `git lfs install`.
  - **Linux (e.g. Ubuntu):** `sudo apt install git-lfs` then `git lfs install`.
- **Flow:** The app can detect size, suggest LFS for &gt;25MB, and handle LFS init, track, add, commit and push.

### GitHub Token

1. Log in to GitHub → profile → **Settings**.
2. **Developer settings** → **Personal access tokens** → **Tokens (classic)**.
3. **Generate new token (classic)**; grant at least `repo`.
4. Copy the token once (it is shown only once).

### Upload notes

- **Git:** Required for upload; install Git (e.g. Git for Windows) if missing.
- **Git LFS:** Required for files &gt;25MB.
- **Network:** Stable access to GitHub needed.
- **Limits:** Single file &lt;100MB on GitHub; &gt;25MB use LFS; free LFS storage is limited (e.g. 1GB).
- **Permissions:** You need write access to the target repo.
- **Paths:** Models go under `models/`, data under `data/` in the repo.

---

## Collaboration tab: GitHub sharing

### Overview

The **Collaboration** tab provides:

1. **Upload:** Training/validation data, trained models, prediction results to GitHub.
2. **Download:** Fetch shared resources from the repository.
3. **Connection:** Configure repo and Token, check status.

### Implementation (for developers)

- **Collaboration tab UI:** `layers/ui_layer/collaboration_tab.py`
- **GitHub adapter:** `layers/collaboration_layer/github_adapter.py`
- **Upload dialog (Help menu):** `layers/ui_layer/upload_dialog.py`
- **Main window:** `GeoDLFWD.py` (Help menu and tab integration)

To add the Collaboration tab in code: import `CollaborationTab` from `layers.ui_layer.collaboration_tab`, create `CollaborationTab(self)`, and add it with `self.tabs.addTab(..., "Collaboration")` (or the desired label).

### Connection

- **Repository owner:** Default "Yuan-Chongxin".
- **Repository name:** Default "GeoDLFWD".
- **Token:** GitHub Personal Access Token with `repo`.
- **Connect:** Click to connect and initialize.

### Upload (from Collaboration tab)

- **Type:** Training data, validation data, model, prediction result, etc.
- **File/folder:** Use "Select file/folder" to choose what to upload.
- **Commit message:** Optional.
- **Upload:** Run upload and watch the log.

### Download

- **Resource list:** Shows available resources from the repo.
- **Refresh:** Fetches the latest list.
- **Download:** Download selected resources to a chosen local path.

### Log

- Operations are logged in real time; errors can be shown in red.

### Workflow

1. **First use:** Open Collaboration tab → enter Token → **Connect to GitHub**.
2. **Upload:** Choose type → select file/folder → (optional) commit message → **Upload to GitHub**.
3. **Download:** **Refresh resource list** → select resources → **Download selected** → choose save path.

### Notes

- Git must be installed for upload/download.
- Stable network required.
- Keep single files under GitHub limits (prefer &lt;100MB; use LFS for &gt;25MB).
- Ensure write permission to the repository.

---

## Troubleshooting

| Issue | Action |
|-------|--------|
| **Git not found** | Install Git (e.g. Git for Windows or GitHub Desktop). |
| **Git LFS not installed** | Install Git LFS and run `git lfs install` (see above). |
| **Connection failed** | Check Token, network, and Token scope (`repo`). |
| **Upload failed** | Check file size, repo write permission, and the operation log. |
| **LFS push failed** | Confirm Git LFS is installed and repo has LFS enabled; check network. |
| **Download failed** | Check resource path and network. |

---

## License

This project is under the MIT License. See [LICENSE](LICENSE).
