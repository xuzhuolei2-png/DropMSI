# DropMSI

DropMSI is a robust deep learning-based framework specifically developed for mass spectrometry imaging (MSI) data imputation. High spatial-resolution MSI data often suffer from severe technical dropouts due to limited detection sensitivity, which may compromise downstream spatial metabolomics analysis. DropMSI addresses this issue by distinguishing technical dropouts from true biological absences and then imputing only the identified dropout signals.
The core concept of DropMSI is to construct a two-stage imputation framework. In the first stage, a fully connected neural network is trained using a semi-synthetic masking strategy to discriminate technical dropouts from biological absences. In the second stage, an uncertainty-aware graph attention network is used to recover missing signals at the identified dropout positions while preserving observed signals and biological absence regions. Developer is Zhuolei Xu from Fuzhou University of China.
# Overview of DropMSI
<div align=center>
<img width="2035" height="1909" alt="DropMSI模型图" src="https://github.com/user-attachments/assets/f9f92093-e541-4517-a454-095ac3a26713" />
<br/>
</div>
Workflow of the proposed DropMSI framework for MSI data imputation. DropMSI consists of two sequential stages. First, a semi-synthetic simulation strategy is used to generate training labels for missing events. Spatially coherent regions with high missing-value rates are identified as candidate biological absences, while observed nonzero signals are randomly masked to simulate technical dropouts. Based on these labels, a three-layer fully connected neural network is optimized to distinguish technical dropouts from biological absences in the original MSI data.
Second, DropMSI performs targeted signal imputation using an uncertainty-aware graph attention network. The MSI pixel grid is represented as a spatial graph, where each pixel is treated as a node and its local spatial neighbors are connected by graph edges. The graph attention network learns spatial-spectral representations from the original MSI data and predicts both the imputed intensity values and the corresponding uncertainty. By introducing uncertainty-guided training, unreliable observed signals are adaptively down-weighted during model optimization. Finally, only the entries identified as technical dropouts are replaced with the reconstructed values, while original observed signals and biological absences are preserved.


# Requirement
    python == 3.9.23
    conda install pytorch=2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
    conda install numpy=1.26.4 pandas=1.5.3 scipy=1.11.4 scikit-learn=1.6.1 matplotlib=3.7.1 tqdm=4.66.1 umap-learn=0.5.9.post2 -c conda-forge
    pip install torch-geometric==2.6.1
    pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

    
# Quickly start
## Input
The input of DropMSI consists of preliminarily preprocessed MSI data with a two-dimensional shape of \[XY, L\], where X and Y represent the pixel numbers for the horizontal and vertical coordinates of the MSI data, respectively, and L represents the number of detected ion channels. Each row corresponds to one spatial pixel, and each column corresponds to one m/z ion feature. Additionally, the input of DropMSI includes an observation-position matrix with a shape of \[XY, L\], a background matrix with a shape of \[XY, 1\], and an imputation-control matrix with a shape of \[XY, L\].

## Run DropMSI model

cd to the DropMSI folder,

if you want to impute mouse kidney MSI data, you can run:

```bash
python run.py \
    --input_NPZ data/kidney/data_132_205.npz \
    --input_Shape 132 205 \
    --processed_Dir proprecessed_data/kidney \
    --output_Dir results/kidney/DropMSI \
    --output_File DropMSI.csv \
    --gpu_ID 0
```

if you want to impute high-resolution mouse hippocampus MSI data acquired at 10 μm spatial resolution, you can run:

```bash
python run.py \
    --input_NPZ data/brain10um/brain10um_162_207.npz \
    --input_Shape 162 207 \
    --processed_Dir proprecessed_data/brain10um \
    --output_Dir results/brain10um/DropMSI \
    --output_File DropMSI.csv \
    --gpu_ID 0
```

if you want to impute mouse HCC MSI data, you can run:

```bash
python run.py \
    --input_NPZ data/HCC/HCC_711_1658.npz \
    --input_Shape 711 1658 \
    --processed_Dir proprecessed_data/HCC \
    --output_Dir results/HCC/DropMSI \
    --output_File DropMSI.csv \
    --gpu_ID 0
```

# Contact

Please contact me if you have any help: xuzhuolei2@gmail.com
