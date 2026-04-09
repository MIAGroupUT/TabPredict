"""
Multi-Modal Data Loaders for EVAR Risk Prediction

Data Sources:
    - Clinical parameters: SPSS/CSV exports from RADAR
    - Imaging: CTA DICOMs stored on Health-RI XNAT
    - Segmentations: Automatic AAA segmentation outputs
    - Geometries: Triangular meshes for geometric deep learning

Classes to implement:
    - EVARDataset: Main dataset class combining clinical + imaging data
    - MultiModalDataLoader: Factory for train/val/test DataLoaders
    - ClinicalDataLoader: SPSS/clinical data only
    - ImagingDataLoader: CTA volume loading with windowing
"""
