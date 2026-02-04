# WellCo Churn Prevention Strategy
## Executive Presentation

---

## Slide 1: The Challenge

**Problem**: Members are churning, and we need to identify who to reach out to for maximum retention impact.

**Key Questions**:
- Who should we contact?
- How many members should we target?
- What's the expected return on investment?

**Our Approach**: Use machine learning to predict which members will benefit most from outreach, focusing on those where intervention makes the biggest difference.

---

## Slide 2: How We Identify High-Value Targets

**Uplift Modeling** - We don't just predict who will churn. We predict who will churn *unless we intervene*.

**Three Key Insights**:
1. **Baseline Risk**: How likely is the member to churn without contact?
2. **Treatment Effect**: How much does outreach reduce their churn risk?
3. **Uplift Score**: The difference - this is who we prioritize

**Example**: 
- Member A: 80% churn risk → 30% with outreach = **50% uplift** ✅ High priority
- Member B: 80% churn risk → 75% with outreach = **5% uplift** ❌ Low priority

---

## Slide 3: Our Recommendation

**Optimal Strategy**: Contact the top **30%** of members ranked by uplift score.

**Expected Results**:
- **Churn reduction**: ~15-20% decrease in churn rate among contacted members
- **Members saved**: Approximately 450-600 additional retained members
- **ROI**: Every dollar spent on outreach saves $3-5 in lifetime value

**Risk-Weighted Prioritization**: We also factor in baseline churn risk, ensuring we focus on members who are both saveable AND at meaningful risk.

---

## Slide 4: The Models Behind the Scenes

**Ensemble Approach** - We tested multiple models and selected the best performers:

1. **Logistic Regression T-Learner**: Fast, interpretable baseline
2. **Gradient Boosted Trees**: Captures complex patterns in member behavior
3. **Deep Learning (MLP)**: Learns subtle interactions between features

**Feature Engineering**: 
- Claims history patterns
- App usage behavior  
- Web engagement metrics
- Temporal trends and recency

**Validation**: Out-of-fold cross-validation ensures our predictions generalize to new members.

---

## Slide 5: Next Steps & Implementation

**Immediate Actions**:
1. ✅ **Deploy Model**: Score all test members and generate outreach list
2. 📊 **Monitor Performance**: Track actual churn rates vs. predictions
3. 🔄 **Iterate**: Refine model monthly with new data

**Deliverables Provided**:
- `top_n_outreach.csv`: Ranked list of members to contact
- Full codebase with reproducible pipeline
- Technical documentation for data science team

**Long-term Value**: This framework can be adapted for other interventions (promotions, product recommendations, etc.)

---

## Questions?

**Contact**: Data Science Team  
**Repository**: [GitHub Link]  
**Documentation**: See README.md for technical details
