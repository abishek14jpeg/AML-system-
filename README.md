# AML-System: Breaking Bad-Inspired Anti-Money Laundering Detection

Inspired by Saul Goodman's iconic explanation of money laundering to Jesse Pinkman in *Breaking Bad* (Season 2, Episode 5: "Breakage"), where he breaks down turning "dirty" cash into legit funds via cash-heavy businesses like nail salons or car washes—mixing illicit proceeds with real revenue to fool the IRS—this project flips the script.

Here, we build an **Anti-Money Laundering (AML) system** to *detect* such schemes algorithmically. Using synthetic transaction data (e.g., PaySim-style datasets), graph neural networks, and anomaly detection, it flags suspicious patterns like smurfing, inflated receipts, or TBML (trade-based money laundering) akin to Walt's car wash ploy or Gus Fring's fronts.

## Key Features
- Transaction network simulation and exploration (built on AMLsim extensions).
- Federated learning for privacy-preserving bank-wide detection.
- GNN models to spot layered laundering tactics.
- Explainable AI (SHAP values) for forensic insights—perfect for crime prevention apps.

Novel twist for academia: Integrates blockchain for immutable audit trails, countering real-world laundering via crypto mixers.

Check out the Saul-Jesse clip for context: [YouTube link](https://www.youtube.com/watch?v=RhsUHDJ0BFM).
