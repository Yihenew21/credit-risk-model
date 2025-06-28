# Credit Scoring Business Understanding

### How does the Basel II Accord's emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord emphasizes accurate risk measurement and transparency to ensure financial institutions maintain adequate capital for credit risk. An interpretable model, such as Logistic Regression with Weight of Evidence (WoE), allows regulators and stakeholders to understand how predictions are made, ensuring compliance with Basel II's supervisory review process. Well-documented models facilitate audits and validation, reducing regulatory risks and ensuring alignment with capital adequacy requirements.

### Why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

Since the dataset lacks a direct "default" label, a proxy variable is necessary to categorize customers as high or low risk. This proxy approximates default likelihood based on behavioral patterns. For this challenge, the key innovation lies in transforming behavioral data into a predictive risk signal by analyzing customer Recency, Frequency, and Monetary (RFM) patterns. This allows for the training of a model that outputs a risk probability score, a vital metric that can be used to inform loan approvals and terms.

However, making predictions based on this proxy carries potential business risks:

- **Misclassification**: The proxy may not accurately reflect true default risk, leading to incorrect loan approvals or rejections.
- **Bias**: If RFM metrics are skewed or incomplete, the model may unfairly penalize certain customer segments.
- **Regulatory Scrutiny**: Regulators may question the validity of the proxy, requiring robust justification and validation.

### What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

- **Simple Models (Logistic Regression with WoE)**:
  - **Pros**: Highly interpretable, easier to explain to regulators, aligns with Basel II's transparency requirements, and is computationally efficient.
  - **Cons**: May have lower predictive power, potentially missing complex patterns in the data.
- **Complex Models (Gradient Boosting)**:
  - **Pros**: Higher predictive accuracy, captures non-linear relationships and complex patterns.
  - **Cons**: Less interpretable, challenging to justify to regulators, higher computational cost, and risk of overfitting.
    In a regulated financial context, interpretability often outweighs marginal performance gains, making a simple, well-understood model like Logistic Regression with WoE a preferred choice unless a complex model can be rigorously validated and explained to regulators.
