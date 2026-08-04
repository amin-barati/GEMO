# **GEMO: A deep learning method for brain fiber classification and tract segmentation using geometrical and morphological features**

PyTorch implementation of GEMO, a deep learning framework for white matter streamline classification and tract segmentation using convolutional neural networks together with handcrafted geometric and morphological features.

**Overview**

Diffusion-weighted magnetic resonance imaging (dMRI) is often used to study brain structure. One of the important applications made possible by dMRI is streamline tractography that is utilized for structural connectivity evaluation and neurosurgical planning. GEMO is a supervised deep learning framework for white matter streamline classification that combines convolutional neural network (CNN) features with handcrafted geometric and morphological descriptors to improve classification performance. This approach benefits from GEometrical and MOrphological features in addition to the features extracted from a convolutional neural network for improving the final classification performance. Streamlines are transformed from three-dimensional coordinate space into two-dimensional color-encoded images using the “xyz2RGB” mapping method and are subsequently fed into a convolutional neural network for learning and classification. This method performs the streamline classification from a whole brain tractogram, only focusing on each streamline’s features, which include those provided from CNN in addition to geometric and morphologic features.

![alt text](Figures/Graphical_abstract.png)

**Complete list of output classes in GEMO**

- AF_L — Left Arcuate Fasciculus
- AF_R — Right Arcuate Fasciculus
- CC_Fr_1 — Corpus Callosum, Frontal Region 1
- CC_Fr_2 — Corpus Callosum, Frontal Region 2
- CC_Oc — Corpus Callosum, Occipital Region
- CC_Pa — Corpus Callosum, Parietal Region
- CC_Pr_Po — Corpus Callosum, Precentral/Postcentral Region
- CG_L — Left Cingulum
- CG_R — Right Cingulum
- FAT_L — Left Frontal Aslant Tract
- FAT_R — Right Frontal Aslant Tract
- FPT_L — Left Frontopontine Tract
- FPT_R — Right Frontopontine Tract
- IFOF_L — Left Inferior Fronto-Occipital Fasciculus
- IFOF_R — Right Inferior Fronto-Occipital Fasciculus
- ILF_L — Left Inferior Longitudinal Fasciculus
- ILF_R — Right Inferior Longitudinal Fasciculus
- MCP — Middle Cerebellar Peduncle
- MdLF_L — Left Middle Longitudinal Fasciculus
- MdLF_R — Right Middle Longitudinal Fasciculus
- POPT_L — Left Parieto-Occipital Pontine Tract
- POPT_R — Right Parieto-Occipital Pontine Tract
- PYT_L — Left Pyramidal (Corticospinal) Tract
- PYT_R — Right Pyramidal (Corticospinal) Tract
- SLF_L — Left Superior Longitudinal Fasciculus
- SLF_R — Right Superior Longitudinal Fasciculus
- UF_L — Left Uncinate Fasciculus
- UF_R — Right Uncinate Fasciculus
- OR_ML_L — Left Optic Radiation (Meyer's Loop)
- OR_ML_R — Right Optic Radiation (Meyer's Loop)

# xyz2RGB

The `xyz2RGB` module converts each streamline of a tractogram into a RGB image by mapping the normalized x, y, and z coordinates to the red, green, and blue color channels, respectively. These images provide a compact representation of streamline geometry and serve as the input to the convolutional neural network (CNN) used in GEMO. The following example converts all streamlines contained in a single `.trk` file into RGB images.

![image_xyz2RGB](Figures/xyz2RGB.png)


**Generate RGB images from a single tractogram**


```python
from xyz2RGB import trk_to_images

images = trk_to_images("Sample_Tract.trk", output_dir="output_images")
```

**Generate RGB images for all tractograms in a directory**

If multiple tractogram files are available, the `process_trk_directory` function automatically processes every `.trk` file in the specified directory. For each tractogram, RGB images are generated for all streamlines and saved to the corresponding output directory while preserving the dataset structure. This function is intended for preparing large datasets prior to feature extraction and network training.

```python
from xyz2RGB import process_trk_directory

results = process_trk_directory("TRK_directory", "Image_directory")
```
# Feature extraction


**Extract geometric and morphological features**

In addition to the RGB images, GEMO utilizes several handcrafted geometric and morphological descriptors for each streamline, including length, curvature, tortuosity, spectral entropy, fractal dimension, and lacunarity. These features are computed once and stored in an HDF5 (`.h5`) file, allowing efficient loading during training without repeated feature computation.

```python
from streamline_features import extract_features_from_directory

extract_features_from_directory("TRK_directory", "features.h5")
```

# Training

 After generating the RGB images and extracting the handcrafted features, the model can be trained using the provided training script. During training, the RGB images are processed by a CNN, the handcrafted features are encoded using a multilayer perceptron (MLP), and the learned representations are fused to classify each streamline into its corresponding white matter tract. The command below starts the complete training pipeline.

 ```bash
 python train.py --trk-dir TRK_directory --features-h5 features.h5 --bounds-h5 bounds.h5
 ```

    GEMO/
    ├── train.py
    ├── streamline_model.py
    ├── dataset.py
    ├── streamline_features.py
    ├── xyz2RGB.py
    ├── utils.py
    ├── config.py    
    ├── requirements.txt
    ├── TRK_directory/
    │   ├── Tract_01.trk
    │   ├── ...
    │   └── Tract_999.trk
    └── features.h5


# Requirements
    python==3.12
    numpy==2.2.6
    scipy==1.16.0
    torch==2.8.0
    torchvision==0.23.0
    nibabel==5.3.2
    h5py==3.14.0
    tqdm==4.67.1
    scikit-learn==1.7.1
    matplotlib==3.10.5

```bash
pip install -r requirements.txt
```

# Citation
GEMO is the code for the following paper; if you use this repository in your research, please cite:

    GEMO: A deep learning method for brain fiber classification and tract segmentation using geometrical and morphological features
    
    Journal: Academic Radiology
    
    DOI:https://doi.org/10.1016/j.acra.2026.07.024