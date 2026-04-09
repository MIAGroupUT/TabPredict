"""
Preprocessing Pipeline for EVAR Data

Processing Steps:
    1. DICOM to NIfTI conversion
    2. Automatic AAA segmentation (nnUNet-based)
    3. Mesh generation from segmentations
    4. CFD feature extraction (thrombus, biomechanics)
    5. Multi-modal fusion and tensor storage

Classes to implement:
    - SegmentationPipeline: nnUNet-based AAA segmentation
    - MeshGenerator: Triangular mesh from binary masks (marching cubes)
    - FeatureExtractor: Per-vertex geometric/biomechanical features
    - DataCleaner: RADAR data validation and cleaning
"""
