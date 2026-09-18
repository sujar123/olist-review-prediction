# Olist Review Prediction

ML pipeline predicting negative customer reviews (1-2 stars) on the Olist 
Brazilian e-commerce dataset, based on delivery, pricing, and product features.

Builds on findings from my SQL analysis project (sql-olist-ecommerce-analysis), 
which found delivery time varies significantly by state (~15-29 days average). 
This project tests whether that variance actually predicts customer dissatisfaction.

## Approach
- Merged order, customer, product, and review data from a MySQL relational schema
- Engineered delivery timing, order complexity, and product-quality features
- Trained Logistic Regression and Random Forest classifiers with stratified 
  train/test splitting and class balancing (bad reviews are a minority class)
- Tuned Random Forest hyperparameters via 3-fold GridSearchCV

## Results
- Test ROC-AUC improved from 0.699 (untuned) to 0.780 (tuned)
- Recall on bad reviews improved from 37% to 60%
- Delivery delay and order item count emerged as the two dominant predictors 
  of a negative review — price and freight cost had minimal influence

## Business takeaway
Prioritize delivery SLA improvements in high-variance states, and flag 
multi-item orders for extra fulfillment quality checks.

**Tools:** Python, MySQL, Pandas, scikit-learn
