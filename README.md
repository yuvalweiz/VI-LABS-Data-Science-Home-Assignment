# WellCo Churn Prediction & Outreach Optimization

This project tackles WellCo's member churn problem by building a data-driven system to identify which members should be contacted for retention outreach. The goal isn't just to predict who might churn, but to figure out who will actually benefit from being contacted—and how many people we should reach out to.

## The Problem

WellCo is losing members and wants to reduce churn through targeted outreach. But here's the catch: outreach costs money, and not everyone responds the same way. Some people might stay anyway, some might leave no matter what, and only a specific group will actually be helped by outreach. We need to find that group and determine the optimal number of people to contact.

## What I Built

I created an end-to-end machine learning pipeline that:

1. **Processes multiple data sources** - Combines web visits, app usage, medical claims, and member information
2. **Engineers meaningful features** - Extracts patterns from behavior data that actually predict outcomes
3. **Builds uplift models** - Uses T-learner models to estimate who benefits most from outreach
4. **Optimizes outreach size** - Determines the sweet spot between cost and benefit
5. **Generates predictions** - Produces a ranked list of members to contact

The code is modular and reproducible, so you can run the entire pipeline from scratch or use individual components.

## Quick Start

### Setup

You'll need Python 3.8+ and preferably a GPU for faster training (but CPU works fine too).

```bash
# Clone the repository
git clone <your-repo-url>
cd vi_labs_assignment

# Create a virtual environment (recommended)
python -m venv my_env
source my_env/bin/activate  # On Mac/Linux
# or: my_env\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

The easiest way to see everything in action is through the Jupyter notebook:

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/vi_labs_pipeline.ipynb
# Then run all cells (Cell → Run All)
```

The notebook walks through the entire process step by step, showing you:
- How the data is loaded and what it looks like
- What features are created and why they matter
- How the models are trained and evaluated
- Which model performs best
- How to determine the optimal number of people to contact
- Final predictions on the test set

## Project Structure

```
vi_labs_assignment/
├── data/
│   ├── train/          # Training data (churn_labels, app_usage, web_visits, claims)
│   └── test/           # Test data for final predictions
├── src/                # All the reusable code
│   ├── config.py       # Settings and constants
│   ├── data_loader.py  # Functions to load CSV files
│   ├── feature_engineering.py  # Feature creation logic
│   ├── feature_selection.py    # Feature importance analysis
│   └── modeling.py     # Model training and evaluation
├── notebooks/
│   └── vi_labs_pipeline.ipynb  # Main notebook - start here!
├── outputs/            # Where results get saved
├── requirements.txt    # Python packages needed
└── README.md          # You're reading it
```

## How It Works

### 1. Feature Engineering

I built features from four data sources:

**From Medical Claims:**
- How many claims someone has filed
- What types of conditions they have (diabetes, hypertension, etc.)
- How recently they've used medical services
- A "severity score" based on their conditions

**From App Usage:**
- How often they use the app
- When they use it (including late-night sessions)
- Whether usage is increasing or decreasing over time

**From Web Visits:**
- What pages they visit (health content vs. other)
- How diverse their browsing is
- Engagement with specific health topics (nutrition, exercise, sleep, etc.)

**Cross-Channel:**
- Overall digital engagement across app and web
- How recently they've interacted with any platform
- Combined engagement scores

The key insight: it's not just about how much someone uses the platform, but *how* they use it and whether that behavior is changing.

### 2. The Uplift Model

This is where it gets interesting. Instead of just predicting churn, I built models that predict the *treatment effect* of outreach. This is called uplift modeling or causal inference.

The approach uses T-learners, which means:
- Train one model to predict churn for people who got outreach
- Train another model to predict churn for people who didn't
- The difference between these predictions is the estimated uplift

I implemented three versions:
1. **Logistic Regression** - Simple and interpretable
2. **Gradient Boosting** - More powerful, captures complex patterns
3. **Deep Learning** - Two-head neural network for maximum flexibility

All models use cross-validation to avoid overfitting, and I use inverse propensity weighting (IPW) to get unbiased estimates even though outreach wasn't randomly assigned.

### 3. Determining Optimal N

The big question: how many people should we contact?

I use several approaches:
- **Qini curves** - Show cumulative benefit as we contact more people
- **Kneedle algorithm** - Finds the point of diminishing returns
- **ROI analysis** - Balances cost of outreach against expected benefit

The answer depends on the cost of outreach and the value of retaining a member, but the model gives you the tools to make that decision.

### 4. Model Evaluation

I evaluate models using:
- **AUQC (Area Under Qini Curve)** - Measures how well the model ranks people by uplift
- **Uplift at different percentiles** - Shows performance at 10%, 20%, 30% outreach rates
- **IPW-adjusted estimates** - Accounts for selection bias in who got outreach

The best model is chosen based on out-of-fold AUQC scores.

## Using the Code

### Option 1: Run the Notebook (Recommended)

Just open `notebooks/vi_labs_pipeline.ipynb` and run the cells. Everything is already set up and documented.

### Option 2: Use the Modules Directly

If you want to integrate this into your own workflow:

```python
from src.data_loader import load_csvs
from src.feature_engineering import build_features
from src.modeling import run_oof_uplift_models_X, build_score_variants

# Load your data
df_churn, df_app, df_web, df_claims = load_csvs("data/train/")

# Create features
df_features = build_features(
    df_churn, df_app, df_web, df_claims, 
    web_mode="counts+conc"  # Use all web features
)

# Train models
from src.feature_selection import get_XYT
X, y, t = get_XYT(df_features)
e_hat, oof_tau, oof_p0, oof_p1 = run_oof_uplift_models_X(X, y, t)

# Get predictions for the best model
best_model = "B_boost_tlearner"  # Or whichever performs best
scores = oof_tau[best_model]

# Rank members by uplift score
df_features['uplift_score'] = scores
top_members = df_features.nlargest(1000, 'uplift_score')
```

## Key Decisions & Rationale

**Why T-learners?**
They're simple, effective, and don't require specialized uplift libraries. They also let us use any base model we want.

**Why IPW weighting?**
The outreach wasn't randomly assigned, so we need to adjust for selection bias. IPW gives us unbiased uplift estimates.

**Why multiple models?**
Different models capture different patterns. Logistic regression is interpretable, boosting is powerful, and deep learning can find complex interactions.

**Why cross-validation?**
To avoid overfitting and get honest performance estimates. All predictions are out-of-fold.

**Why these features?**
I focused on behavioral patterns that indicate engagement and health awareness. Someone who's actively using health content is different from someone who just browses randomly.

## Output Files

After running the pipeline, you'll get:
- **Ranked member list** - CSV with member IDs, uplift scores, and ranks
- **Model comparison** - Performance metrics for all models
- **Qini curves** - Visualizations of uplift performance
- **Feature importance** - Which features matter most

## Technical Notes

- The code uses GPU acceleration if available (for CatBoost and PyTorch)
- All random seeds are set for reproducibility
- The pipeline handles missing data automatically
- Feature engineering is vectorized for speed

## What's Next?

To use this in production:
1. Run the pipeline on your latest data
2. Choose your optimal N based on business constraints
3. Export the top N members for outreach
4. Track actual outcomes to validate the model
5. Retrain periodically as new data comes in

## Questions?

The notebook has detailed comments explaining each step. If something's unclear, check there first. The code is modular, so you can also read individual functions in the `src/` folder to understand specific components.

---

**Note:** This was built for the VI Labs data science assignment. The approach is general-purpose and can be adapted to other uplift modeling problems.
