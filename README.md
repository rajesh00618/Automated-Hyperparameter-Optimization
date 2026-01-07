# XGBoost Hyperparameter Optimization with Optuna & MLflow

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-3.8+-orange.svg)](https://mlflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-4.6+-green.svg)](https://optuna.org/)

A production-grade, automated hyperparameter optimization pipeline for XGBoost regression models using Optuna and MLflow. This project demonstrates advanced MLOps practices with parallel optimization, comprehensive experiment tracking, and containerized deployment.

## 🎯 Project Overview

This project implements an end-to-end MLOps workflow that combines automated hyperparameter search with systematic experiment tracking. By leveraging Optuna's intelligent optimization algorithms and MLflow's experiment management, the pipeline efficiently finds optimal XGBoost configurations while maintaining full reproducibility and auditability.

### Key Features

- **Intelligent Hyperparameter Search**: 7-dimensional search space with appropriate distributions
- **Parallel Optimization**: Concurrent trial execution for faster optimization
- **Early Stopping**: Median pruning to eliminate unpromising trials
- **Comprehensive Tracking**: Full MLflow integration with metrics, parameters, and artifacts
- **Containerized Deployment**: Docker-based execution for reproducibility
- **Interactive Analysis**: Jupyter notebook with optimization insights and visualizations
- **Production Ready**: Reproducible results with fixed random seeds

## 📊 Performance Results

The optimized XGBoost model achieves excellent performance on the California Housing dataset:

- **Test RMSE**: 0.451 (target: < 0.75 ✓)
- **Test R²**: 0.845 (target: > 0.70 ✓)
- **Optimization Time**: ~31 seconds (target: < 30 minutes ✓)
- **Trials Completed**: 12+ trials with intelligent pruning

## 🏗️ Architecture

```
xgboost_optuna_project/
├── Dockerfile                 # Containerized deployment
├── Dockerfile.demo           # Demo application container
├── requirements.txt          # Python dependencies
├── requirements_demo.txt     # Demo app dependencies
├── demo_app.py              # Streamlit demo application
├── deploy_demo.sh           # Demo deployment script
├── notebooks/
│   └── analysis.ipynb        # Interactive analysis notebook
├── outputs/                  # Generated results and artifacts
│   ├── results.json         # Optimization summary
│   ├── optimization_history.png
│   ├── param_importance.png
│   └── mlruns/              # MLflow tracking data
└── src/
    ├── __init__.py          # Python package
    ├── data_loader.py       # Data loading utilities
    ├── objective.py         # Optuna objective function
    ├── optimize.py          # Main optimization pipeline
    ├── evaluate.py          # Results generation
    └── optuna_study.db      # Study database (generated)
```

## 🎬 Live Demo & Video Demo

### **Live Interactive Demo**
A Streamlit web application that showcases the optimization results interactively:

**🌐 Live Demo URL**: [Deploy on your preferred cloud platform]

**Features:**
- 📊 Real-time metrics dashboard
- 📈 Interactive optimization history plots
- 🎯 Hyperparameter importance visualization
- 📋 Detailed trial-by-trial results
- 🔍 Performance comparison with baseline

**Local Demo Setup:**
```bash
# Ensure outputs exist from optimization
ls outputs/results.json

# Install demo dependencies
pip install -r requirements_demo.txt

# Run the demo app
streamlit run demo_app.py
```

**Docker Demo Deployment:**
```bash
# Build demo image
docker build -f Dockerfile.demo -t xgboost-demo .

# Run locally
docker run -p 8501:8501 xgboost-demo

# Access at: http://localhost:8501
```

**Cloud Deployment Options:**
- **Heroku**: `heroku create && git push heroku main`
- **Railway**: Connect GitHub repository
- **Render**: Deploy from Docker image
- **AWS/GCP**: Use ECS/EKS or Cloud Run

### **Video Demo**
**📹 Video Demo URL**: [Upload to YouTube/Vimeo/Google Drive]

**Video Content:**
1. **Project Overview** (0:00-1:30)
   - Problem statement and objectives
   - Technologies used

2. **Docker Execution** (1:30-4:00)
   - Building the container
   - Running the optimization pipeline
   - Monitoring progress

3. **Results Analysis** (4:00-7:00)
   - Performance metrics achieved
   - Hyperparameter importance insights
   - MLflow experiment tracking

4. **Live Demo Walkthrough** (7:00-9:00)
   - Interactive dashboard features
   - Visualization explanations

5. **Technical Deep Dive** (9:00-12:00)
   - Code architecture explanation
   - MLOps best practices demonstrated

**Video Specifications:**
- Duration: 10-12 minutes
- Resolution: 1080p
- Include voice narration
- Show terminal commands
- Demonstrate all key features

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized execution)

### Local Execution

1. **Clone and setup**:
   ```bash
   git clone https://github.com/rajesh00618/Automated-Hyperparameter-Optimization.git
   cd Automated-Hyperparameter-Optimization
   pip install -r requirements.txt
   ```

2. **Run optimization**:
   ```bash
   cd src
   python optimize.py
   ```

3. **View results**:
   ```bash
   cd ../notebooks
   jupyter notebook analysis.ipynb
   ```

4. **Launch interactive demo**:
   ```bash
   pip install -r requirements_demo.txt
   streamlit run demo_app.py
   ```

### Docker Execution

1. **Build container**:
   ```bash
   docker build -t xgboost-optuna-pipeline .
   ```

2. **Run optimization**:
   ```bash
   docker run --rm -v $(pwd)/outputs:/app/outputs xgboost-optuna-pipeline
   ```

3. **Deploy demo**:
   ```bash
   # Build demo container
   docker build -f Dockerfile.demo -t xgboost-demo .

   # Run demo locally
   docker run -p 8501:8501 xgboost-demo
   ```

4. **Access interfaces**:
   - **Demo App**: http://localhost:8501
   - **Results**: `outputs/results.json`
   - **Visualizations**: `outputs/*.png`
   - **MLflow UI**: `mlflow ui --backend-store-uri outputs/mlruns`

## ☁️ Cloud Deployment Options

### **Heroku Deployment**
```bash
# Install Heroku CLI and login
heroku create your-xgboost-demo

# Deploy demo app
git push heroku main

# Access at: https://your-xgboost-demo.herokuapp.com
```

### **Railway Deployment**
1. Connect your GitHub repository to Railway
2. Set build command: `docker build -f Dockerfile.demo -t app .`
3. Set start command: `streamlit run demo_app.py --server.port=$PORT --server.address=0.0.0.0`
4. Deploy automatically on git push

### **Render Deployment**
1. Create new Web Service from Docker
2. Connect GitHub repository
3. Set Docker command: `docker build -f Dockerfile.demo -t app .`
4. Deploy

### **AWS/GCP Deployment**
```bash
# Build and push to container registry
docker build -f Dockerfile.demo -t xgboost-demo .
docker tag xgboost-demo gcr.io/your-project/xgboost-demo
docker push gcr.io/your-project/xgboost-demo

# Deploy to Cloud Run (GCP) or ECS (AWS)
```

### **Local Development Server**
```bash
# Using the deployment script
chmod +x deploy_demo.sh
./deploy_demo.sh
```

## 🔧 Configuration

### Hyperparameter Search Space

| Parameter | Range | Distribution | Description |
|-----------|-------|--------------|-------------|
| `n_estimators` | 50-300 | Integer | Number of boosting rounds |
| `max_depth` | 3-10 | Integer | Maximum tree depth |
| `learning_rate` | 0.001-0.3 | Log Uniform | Step size shrinkage |
| `subsample` | 0.6-1.0 | Uniform | Subsample ratio of training instances |
| `colsample_bytree` | 0.6-1.0 | Uniform | Subsample ratio of columns |
| `min_child_weight` | 1-10 | Integer | Minimum sum of instance weight |
| `gamma` | 0-0.5 | Uniform | Minimum loss reduction |

### Optimization Settings

- **Trials**: 100 optimization trials
- **Cross-validation**: 5-fold CV within each trial
- **Pruning**: MedianPruner (n_startup_trials=10, n_warmup_steps=5)
- **Parallelization**: 2 concurrent jobs
- **Objective**: Minimize negative mean squared error
- **Storage**: SQLite database backend

## 📈 Analysis & Insights

The analysis notebook provides comprehensive insights:

### Optimization History
- Tracks objective value improvement over trials
- Shows pruning effectiveness
- Identifies convergence patterns

### Parameter Importance
- Quantifies each hyperparameter's impact
- Guides future optimization efforts
- Reveals parameter interactions

### Baseline Comparison
- Compares optimized vs. default XGBoost
- Quantifies performance improvements
- Validates optimization value

### Key Findings
- **Learning Rate**: Most critical parameter (log-scale optimization essential)
- **Tree Depth**: Significant impact on model complexity
- **Regularization**: Gamma and min_child_weight provide fine-tuning
- **Sampling Ratios**: Subsample and colsample_bytree offer robustness

## 🛠️ Technical Implementation

### Core Components

#### `data.py`
```python
def load_and_split_data(test_size=0.2, random_state=42):
    """Load California Housing dataset and split into train/test sets."""
```

#### `objective.py`
```python
def objective(trial, X_train, y_train):
    """Optuna objective function defining hyperparameter search space."""
```

#### `optimize.py`
```python
# Main pipeline orchestrating:
# 1. Data loading
# 2. Optuna study setup
# 3. Parallel optimization
# 4. MLflow logging
# 5. Best model training and evaluation
# 6. Results visualization and export
```

### MLflow Integration

- **Experiment**: `optuna-xgboost-optimization`
- **Parameters**: All 7 hyperparameters logged per trial
- **Metrics**: CV MSE/RMSE, trial metadata
- **Tags**: Trial state (COMPLETE/PRUNED/FAIL), best model flag
- **Artifacts**: Trained model, visualizations

### Reproducibility

- Fixed random seeds: `random.seed(42)`, `np.random.seed(42)`
- Deterministic train/test splits
- Versioned dependencies
- Containerized execution

## 📊 Output Files

### `results.json`
```json
{
  "n_trials_completed": 12,
  "n_trials_pruned": 0,
  "best_cv_rmse": 0.458,
  "test_rmse": 0.451,
  "test_r2": 0.845,
  "best_params": {...},
  "optimization_time_seconds": 31.39
}
```

### Visualizations
- `optimization_history.png`: Trial progression
- `param_importance.png`: Feature importance ranking

### Database
- `optuna_study.db`: Complete optimization history
- `mlruns/`: MLflow experiment tracking

## 🔍 Advanced Usage

### Customizing Search Space

Modify `objective.py` to adjust hyperparameter ranges:

```python
# Example: Narrow learning rate range
"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
```

### Different Datasets

Adapt `data.py` for custom datasets:

```python
def load_and_split_data():
    # Your custom data loading logic
    X, y = load_your_data()
    return train_test_split(X, y, test_size=0.2, random_state=42)
```

### Scaling Optimization

For larger search spaces or longer runs:

```python
study.optimize(..., n_trials=500, n_jobs=4)  # More trials, more parallelism
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Optuna](https://optuna.org/) for intelligent hyperparameter optimization
- [MLflow](https://mlflow.org/) for experiment tracking
- [XGBoost](https://xgboost.readthedocs.io/) for gradient boosting
- [California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the analysis notebook for detailed insights
- Review MLflow logs for debugging

---

**Built with  for the ML engineering community**