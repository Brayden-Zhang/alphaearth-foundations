# Extending AEF for Dataset Generation

Link: https://arxiv.org/pdf/2508.11739


Problem: High-quality labeled geospatial datasets (e.g., land cover, vegetation type) are sparse and often restricted to certain regions (e.g., LANDFIRE’s vegetation type dataset is USA-only).

Solution: Use AEF embeddings (64-dim, global, multi-sensor, annual, 10m resolution) as input features, then train simple supervised models (RF, GBT, U-Net segmentation, etc.) on existing labels in one region (USA), and extend those labels to new unlabeled regions (Canada).

Dataset: LANDFIRE's vegetation type dataset (EVT), which is used for wildfire modeling.

Training Region: Alaska + Northern CONUS (above ~41.6°N).

Test Region: Canadian southern/western border zones (90 km wide bands).

Models: Logistic regression, Random Forest, Gradient Boosted Trees (LightGBM), U-Net segmentation.

Similar works: See applications of the Clay foundation model for similar use cases.