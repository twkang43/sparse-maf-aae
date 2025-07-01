# Sparse MAF-AAE
This is the official repository for the paper "**[A real-time anomaly detection method for robots based on a flexible and sparse latent space](https://arxiv.org/abs/2504.11170)**", publised in *[Engineering Applications of Artificial Intelligence](https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence)*.

## Get Started
### 1. Clone the Repository
To download the codes, please clone this repository.
```
git clone https://github.com/twkang43/Sparse-MAF-AAE.git
cd sparse-maf-aae
```

### 2. Configure Python Environments
The code requires a `python>=3.11` environments. Follow the instruction.<br/>
We recommend to use conda:
```
conda create -n <env_name> python=3.11
conda activate <env_name>
pip install -r requirements.txt 
```

### 3. Download the Dataset
We used the voruas-AD dataset from the paper titled "[The voraus-AD Dataset for Anomaly Detection in Robot Applications](https://ieeexplore.ieee.org/abstract/document/10315239)" by Jan Thieß Brockmann, Marco Rudolph, Bodo Rosenhahn, and Bastian Wandt.

To download the dataset, please refer to the instructions provided in the [official Github repository](https://github.com/vorausrobotik/voraus-ad-dataset) of the voraus-AD.

We employed the 100 Hz dataset for our implementation.
To reproduce our results, please make sure to use the 100 Hz dataset as well.

The voraus-AD dataset should be placed in the following path:
```
<your_path_to_the_code>/dataset/voraus-AD/
```

## Running Sparse MAF-AAE
You can run Sprase MAF-AAE with the following instructions:
```
python main.py \
  --exec <exec> \
  --signals <signals> \
  --frequency <frequency> \
  --subset_size <subset_size>
```

There are four arguments:

- `exec`: This determines the execution mode of the code. The choices are `train`, `eval`, and `all`. If you use `all`, the code will first train the model and then automatically evaluate it after training is completed. The defualt setting is `all`.

- `signals`: This specifies the type of signals to use. The choices available are `torques`, `currents`, `encoders`, and `machine`. The default option is `torques`.

- `frequency`: The parameter denotes the frequency of the dataset. The available frequencies are 100 Hz, 50 Hz, 25 Hz, and 10 Hz. The default value is 50 Hz.

- `subset_size`: This indicates the size of the subset used for training (train gain). The available choices for this parameter are 1.0 (948), 0.5 (474), 0.25 (237), and 0.125 (118). The default value is 1.0.

## Main Result
We compared our model with 3 baselines: LSTM-VAE, Anomaly Transformer, and MVT-Flow.

Generally, Sparse MAF-AAE achieves SOTA on various scenarios.

<img src="https://lh3.googleusercontent.com/fife/ALs6j_FXH54FKYZTbVFA1suekJqma6NcZQrA3CQl9SO1Cpe5pXnayGDG0T1xWyzuRHmOnEkVowlhQtHOaBontAdgf5sbycG6o2bkgkcB-aatGPkJbD61a2z16lZc026K1GzuPrWe3FFNyc29Tz__XV_sdxPTnnFCARtv-KGZj6ccqPyqQUw4l8aNS7cznw9KS1zcGlmK7kHM7wZ-9B0V4Xb9bESBnlgJXGF1bF6FIB8pV9FcnlYCAERlM0fGnIq74iw6HB0FtWjroYAbd1hT7BWIyq7DKGC6uablhPFN_TXjfMTY-CsGnDGqQcEqrLgAaLVLt9mDCPdxPO2o72qPJ0iiaIPYPHo-cXRrpExAze4phi3LLj5CDyswAz7jtu7o3VjN4rJ5RTdd13Lc82x7TbbIhRVddwbjl4iNcdeBgfqsoqOKaSDb0teAWJtFd1LrcP-OtVeUtba2c5-SifG2Gvz72crhcbtzOWaxKPtPSYluXcvVuhEITXCS2YbCQMmiZx0llA82k30oMTdq23xgjLOeZowMXr7ksH4a_HI3xzr0GU90LM8Gl-cONgnXGSZTlrK6Rk4lQlUHyL_GLHHCFkgD1V9k3JzE_Q6giP709g4H-yjejydCynw3YgU838D_w_5DkD2oQkToTtk0E8Z3QBVMoUET4Zs0cOoQTvwe3_jF8hWaG4fXzVmTG9HmnNgHgf3lgACVHGPBDLaUQ66ILFwhHqvBwweVFpFy0M6BAJfUarzCOSaF2gT_w3ddgb5aDGUMDg43CUaFID7c57cuwSzRGzQCDSVzrQBjFa6ZTF8XrCH3pZFKBBpcBQcShcjBDDuRTVWLGB8eBuFkvAIxX40Ptu14-eTMGVSOEbxWZp4x5g27JCCnNWqTKvGIfVF37IUR5-ZQlOGIVIbVrHvC4W4GZs1NLN3o2xr9Z1au_FKB2akGrkHM5pF2HPm9CJu1aqWc3e-OVSEWDLEnKxgnOzKVp2-98t33NGbK1353HpKgGYou68TiHdeSu0mqKCcqWqCPOvuAXuG8T_zXtgxRWb-AIU4HsQQHFgAaliWiMDVt2Nib2F1t_PfLU0C3CtVgi6XgJV8NgforjcMsc7_koZ6LlQDLFUtHxpRP5tjcL-x7QCEo_laETnA5OJnuueCaqGV9sC1ccLdV3fYXJtvfXS5q9rG6PNcjWFYs_Srv_hJRZED2N3SbXPYTy1bAlAFGrJspYbVFD9beU98rtqU80bIqIadhiDBGS5gerb5zVrBuqu-7mU7uxrTm9iwSvVgIRvAsWNJ2ioJtY15wmdKQVKduaKD_J3aIGgZP0eVpy34evaOek6ru_PU64NC-n4LXXtGRLmHzZN0QbLML6WnLbm7U_kgr6tuip_mx_gh4ncdn6jsGXT8KVrcFbBlpan81ezw2U-EWJDMtJpZivEPHd23_P7HOMtQ4a31jwt9jGlGT_2T1TKeJH5N8BFF093S385j7HuZL_-KEBDbTS4PelXRG0BpRtZZhW94674pmHxDybkJfAcf3bcPT7m4ZFSLj1YLaXxOfMO4BgAOA_VxHhtLyFN3bgC3ulrAha0VyICioVTaZQcsGtMONIKodZPPqMxNNcQD6bfLTnt9W0A3IMIwqIXl91Qjo3iLna6dilwJxar2KTsMnMgCKLtVW5wqBldqMbzO-ISEi0Fa4-MlAfml9N8Ur6FNd8g=w2880-h1500" alt="Sparse MAF-AAE main results" style="width:60%; height:auto;" />

## Citation
If you use the Sparse MAF-AAE in your research, please cite the following paper:
```
@article{kang2025real,
  title={A real-time anomaly detection method for robots based on a flexible and sparse latent space},
  author={Kang, Taewook and You, Bum-Jae and Park, Juyoun and Lee, Yisoo},
  journal={Engineering Applications of Artificial Intelligence},
  volume={158},
  pages={111310},
  year={2025},
  publisher={Elsevier}
}
```

## License 
This project is licensed under the [MIT License](./LICENSE).
