# AI for EVAR: Optimizing Treatment and Follow-up

## Overview
This project develops AI-based risk prediction models for patients undergoing **EndoVascular Aneurysm Repair (EVAR)**. The goal is to personalize treatment planning and follow-up frequency by predicting complications and re-interventions using multi-modal data.

## Work Packages
* **WP1:** Development of pre- and post-operative AI models for re-intervention prediction.
* **WP2:** Integration of models into Shared Decision Making (SDM) tools (3D prints, holograms, 3mensio).
* **WP3:** Early Health Technology Assessment (HTA).

## Data Source
We utilize the **RADAR database**, containing clinical and imaging data from over 1,000 AAA patients.

## Technical Stack
- **Imaging:** CT/CTA scans with automatic segmentation.
- **AI:** Geometric Deep Learning operating on triangular meshes.
- **Clinical Data:** Tabular data formatted via an established dictionary.

## Installation
1. Clone the repo: `git clone https://github.com/your-org/ai-for-evar.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Configure your local paths in `config.yaml`.
