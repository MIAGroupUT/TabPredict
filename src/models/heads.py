"""
Prediction Heads for EVAR Risk Models

Task-specific output heads for WP1:
    1. Preoperative model: Sac regression and treatment optimization
    2. Postoperative model: Re-intervention risk stratification

Classes to implement:
    - PreoperativeHead: Sac regression, complication risks, graft sizing
    - PostoperativeHead: Re-intervention prob, survival curve, follow-up interval
    - ReinterventionPredictor: End-to-end wrapper (backbone + head)
    - SacRegressionHead: Specialized AAA diameter change prediction
"""
