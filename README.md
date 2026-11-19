# Assignment No. 7: Implement CI/CD Pipeline Using GitHub Actions

## 1. Project Title
**MLOps CI/CD Pipeline for Iris Classification using GitHub Actions**

---

## 2. Project Description
This project implements an automated Continuous Integration and Continuous Deployment (CI/CD) pipeline for a Machine Learning application using GitHub Actions. It trains a Logistic Regression classifier on the classic Iris flower dataset, verifies model accuracy through automated unit tests with `pytest`, and simulates automated deployment once tests successfully pass on the `main` branch.

---

## 3. Aim
To design, implement, and validate an end-to-end CI/CD workflow using GitHub Actions that automates code checkout, Python environment setup, dependency management, unit testing, and deployment triggered by Git push and pull request events.

---

## 4. Technologies Used
- **Language**: Python 3.11
- **Machine Learning Library**: `scikit-learn`
- **Testing Framework**: `pytest`
- **Automation / CI/CD Engine**: GitHub Actions
- **Version Control**: Git & GitHub
- **Runner Environment**: `ubuntu-latest`

---

## 5. Project Structure
```text
mlops-cicd/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # GitHub Actions workflow configuration
│
├── src/
│   ├── __init__.py            # Python package initialization
│   └── model.py               # ML model training and evaluation script
│
├── tests/
│   ├── __init__.py            # Python package initialization
│   └── test_model.py          # Pytest unit tests for model accuracy
│
├── .gitignore                 # Files and folders ignored by Git
├── app.py                     # Application entry point CLI
├── README.md                  # Project documentation and practical guide
└── requirements.txt           # Project dependencies
```

---

## 6. How the ML Model Works
1. **Dataset**: Uses the popular **Iris dataset** containing 150 samples across 3 flower species (*Setosa*, *Versicolor*, *Virginica*) with 4 features (sepal length, sepal width, petal length, petal width).
2. **Data Split**: The dataset is divided using `train_test_split` with `test_size=0.2` (80% training data, 20% testing data) and a fixed `random_state=42` to ensure reproducible splits.
3. **Algorithm**: A **Logistic Regression** classifier is initialized with `max_iter=200` to allow gradient convergence.
4. **Training**: The model learns decision boundaries on the training dataset (`X_train`, `y_train`).
5. **Evaluation**: Predictions are made on `X_test`, and `accuracy_score` is computed against `y_test`.
6. **Return**: The `train_model()` function returns the calculated accuracy score.

---

## 7. How to Install Dependencies
Open a terminal in the project directory (`mlops-cicd/`) and run:

```bash
# Optional: Create and activate a virtual environment
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On Linux / macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 8. How to Execute the Application
Run the standalone model script:
```bash
python src/model.py
```

Run the main application entry point:
```bash
python app.py
```

**Expected Output:**
```text
============================================================
       MLOps CI/CD Demo - Machine Learning Application       
============================================================
Application Name  : Iris Classification App
ML Algorithm      : Logistic Regression (max_iter=200)
Dataset           : Iris Dataset (scikit-learn)
------------------------------------------------------------
Training and evaluating model...
Model Accuracy    : 1.0000 (100.00%)
------------------------------------------------------------
Execution Status  : Application executed successfully!
============================================================
```

---

## 9. How to Execute Unit Tests
Run the test suite using `pytest`:
```bash
pytest -v
```

**Expected Output:**
```text
tests/test_model.py::test_model_accuracy PASSED                      [100%]
============================== 1 passed in ... ==============================
```

The assertion verifies that:
```python
assert accuracy >= 0.80
```
Since the model attains 1.0000 accuracy, the test passes with flying colors.

---

## 10. Explanation of the CI/CD Workflow (`ci-cd.yml`)
The workflow configuration is defined inside `.github/workflows/ci-cd.yml`:

- **Triggers**:
  - `push` to the `main` branch.
  - `pull_request` targeting the `main` branch.
- **Job 1: Test ML Application (`test`)**:
  - Runs on `ubuntu-latest`.
  - Checks out the repository using `actions/checkout@v4`.
  - Configures Python 3.11 with `actions/setup-python@v5`.
  - Upgrades `pip` and installs packages from `requirements.txt`.
  - Executes `pytest -v`.
  - If any unit test fails, this job fails immediately.
- **Job 2: Deploy Application (`deploy`)**:
  - Dependent on testing: uses `needs: test`.
  - Conditional Execution: `if: github.event_name == 'push' && github.ref == 'refs/heads/main'`.
  - Only runs on direct pushes to `main` when the `test` job passes.
  - Automatically skipped on pull requests or if testing fails.
  - Simulates deployment by outputting deployment messages.

---

## 11. Explanation of Continuous Integration (CI)
Continuous Integration (CI) is the practice of automating the integration of code changes from multiple contributors into a shared repository frequently. In our project:
- Every code push or pull request automatically triggers the test pipeline.
- GitHub Actions spins up an Ubuntu runner, installs dependencies, and runs `pytest`.
- Bugs and regression issues are detected immediately before code merges into production.

---

## 12. Explanation of Continuous Deployment (CD)
Continuous Deployment (CD) is the automated release of validated code to production or staging environments without manual intervention. In our project:
- Once the CI test job succeeds on `main`, the `deploy` job triggers automatically.
- Pull requests only run the CI test job, protecting the deployment target from unmerged or untested code.

---

## 13. How to Create and Connect a GitHub Repository
Follow these steps to push this project to GitHub:

1. **Create Repository on GitHub**:
   - Go to [GitHub](https://github.com/) and log in.
   - Click **New** repository.
   - Name it `mlops-cicd`.
   - Leave it **Public** (recommended for GitHub Actions).
   - Do **NOT** initialize with README, .gitignore, or license (they already exist locally).
   - Click **Create repository**.

2. **Connect Local Project to GitHub**:
   In your terminal inside `mlops-cicd`:
   ```bash
   git init
   git branch -M main
   git add .
   git commit -m "feat: complete MLOps Assignment 7 CI/CD pipeline"
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/mlops-cicd.git
   git push -u origin main
   ```

---

## 14. How to View GitHub Actions Results
1. Navigate to your repository on GitHub (`https://github.com/<YOUR_GITHUB_USERNAME>/mlops-cicd`).
2. Click on the **Actions** tab at the top.
3. You will see the latest workflow run: **MLOps CI/CD Pipeline**.
4. Click on the workflow run to view the interactive pipeline graph with both jobs:
   - `Test ML Application` (Green checkmark)
   - `Deploy Application` (Green checkmark)
5. Click on individual jobs to inspect live logs for every step.

---

## 15. How to Demonstrate the Test Failure Scenario (Part F)
To demonstrate that CI properly protects the pipeline and halts deployment when tests fail:

### Step 1: Temporarily break the test
Open `tests/test_model.py` and modify the threshold to an impossible value (e.g. `1.01` or `> 1.0`):
*(Note: Because Iris Logistic Regression achieves 1.0000 accuracy, setting `assert accuracy >= 1.01` or `assert accuracy < 0.80` triggers the expected failure).*

```python
def test_model_accuracy():
    accuracy = train_model()
    # Temporarily changed to demonstrate CI failure
    assert accuracy >= 1.01
```

### Step 2: Test locally
Run `pytest -v`:
```bash
pytest -v
```
You will see `FAILED tests/test_model.py::test_model_accuracy - AssertionError`.

### Step 3: Push failure to GitHub
```bash
git add tests/test_model.py
git commit -m "test: simulate test failure for CI/CD demonstration"
git push origin main
```

### Step 4: Verify on GitHub Actions
- Open the **Actions** tab.
- The `Test ML Application` job will show a red **X** (Failed).
- The `Deploy Application` job will show **Skipped** (deployment blocked!).

### Step 5: Restore and fix
Restore `tests/test_model.py` to:
```python
def test_model_accuracy():
    accuracy = train_model()
    assert accuracy >= 0.80
```

Push the fix:
```bash
git add tests/test_model.py
git commit -m "fix: restore passing test assertion threshold"
git push origin main
```
The workflow will run and both jobs will turn green again!

---

## 16. Required Screenshots Checklist for Practical Report
Capture the following screenshots for your practical submission:
1. **Project Directory Tree**: Showing all created files and folders in your IDE / Terminal.
2. **Local Model & App Execution**: Terminal showing successful run of `python app.py`.
3. **Local Pytest Success**: Terminal showing `pytest -v` passing with green output.
4. **Local Pytest Failure**: Terminal showing `pytest -v` failing when threshold is altered.
5. **GitHub Actions Successful Run**: GitHub browser tab showing green checkmarks for both `test` and `deploy` jobs.
6. **GitHub Actions Failure & Skipped Deployment**: GitHub browser tab showing red failed `test` job and skipped `deploy` job.
