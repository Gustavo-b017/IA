# Logistic Regression Teaching Visuals — Code Bundle

This bundle contains scripts to reproduce every animation/figure we created.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Example:
python 01_logreg_1d_training.py
```
Outputs will be written to the current directory.

## Contents

### 1D demos
- **01_logreg_1d_training.py** → `logistic_regression_training.gif` (1D, sigmoid over x; noisy dataset)  
- **02_logreg_1d_with_zline_scaled.py** → `logistic_regression_with_z.gif` (adds scaled z line)  
- **03_logreg_1d_sigmoid_plus_z_subplots.py** → `logistic_regression_sigmoid_plus_z.gif` (top: sigmoid; bottom: raw z)  
- **04_logreg_1d_separate_views.py** → `sigmoid_probabilities.gif`, `linear_scores.gif` (two separate synced views)  
- **05_logreg_1d_clean_combined_annotated.py** → `logistic_regression_clean_combined_annotated.gif` (clean separable data; arrows + w,b)

### Real-data (Breast Cancer, 2 features)
- **06_breast_cancer_2d_boundary.py** → `breast_cancer_logreg_boundary.gif` (moving 2D decision line)  
- **07_logistic_surface_3d_static.py** → `logistic_surface_3d.png` (3D probability surface + 0.5 contour)  
- **08_logistic_surface_3d_training.py** → `logistic_surface_training_3d.gif` (surface changes during training)  
- **09_logistic_surface_static_and_rotate.py** → several `logistic_surface_view_e*_a*.png` + `logistic_surface_rotate.gif` (camera sweeps)

### Combined views (stitched, no subplots)
- **10_combined_zline_plus_surface.py** → `combined_zline_plus_surface.gif` (2D z-line vs 3D surface)  
- **11_combined_features_by_z_plus_surface.py** → `combined_features_by_z_plus_surface.gif` (2D colored by z + 3D surface)  
- **12_combined_features_by_z_plus_surface_zoomed.py** → `combined_features_by_z_plus_surface_zoomed.gif` (tighter zoom)  
- **13_combined_features_by_z_plus_surface_zoomed2.py** → `combined_features_by_z_plus_surface_zoomed2.gif` (even tighter + new angle)

All scripts accept optional CLI args (see top of each file) for steps, learning rate, padding, camera, etc.
