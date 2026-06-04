#!/bin/bash
# Run from project root: bash setup_github.sh YOUR_GITHUB_USERNAME
# Prerequisites: git installed, GitHub account created, repo created at
# https://github.com/YOUR_USERNAME/nhanes-biological-age

USERNAME=$1
if [ -z "$USERNAME" ]; then
  echo "Usage: bash setup_github.sh YOUR_GITHUB_USERNAME"
  exit 1
fi

echo "Setting up git repo for github.com/$USERNAME/nhanes-biological-age"

git init
git add .
git commit -m "Initial commit: NHANES biological age & mortality predictor

- NHANES 2015-2020 data pipeline (3 cycles, 18,328 participants)
- XGBoost mortality prediction: AUC=0.842
- SHAP feature importance: RDW ranked #2 behind age
- Biological age score (linearly calibrated from xb)
- Inference script for individual blood panel prediction"

git branch -M main
git remote add origin https://github.com/$USERNAME/nhanes-biological-age.git
git push -u origin main

echo ""
echo "Done! Visit: https://github.com/$USERNAME/nhanes-biological-age"
