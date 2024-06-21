# Sparse MAF-AAE
This is the official repository to the paper "Sparse MAF-AAE".

![Sprase MAF-AAE]()

## Get Started
### 1. Clone the Repository
To download the codes, please clone this repository.
```
git clone https://github.com/twkang43/Sparse-MAF-AAE.git
cd Sparse-MAF-AAE
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
python main.py --exec <exec> --signals <signals> --frequency <frequency> --subset_size <subset_size>
```

There are four arguments:

- `exec`: This determines the execution mode of the code. The choices are `train`, `eval`, and `all`. If you use `all`, the code will first train the model and then automatically evaluate it after training is completed. The defualt setting is `all`.

- `signals`: This specifies the type of signals to use. The choices available are `torques`, `currents`, `encoders`, and `machine`. The default option is `torques`.

- `frequency`: The parameter denotes the frequency of the dataset. The available frequencies are 100 Hz, 50 Hz, 25 Hz, and 10 Hz. The default value is 50 Hz.

- `subset_size`: This indicates the size of the subset used for training (train gain). The available choices for this parameter are 1.0 (948), 0.5 (474), 0.25 (237), and 0.125 (118). The default value is 1.0.

## Main Result
We compared our model with 3 baselines: LSTM-VAE, Anomaly Transformer, and MVT-Flow.

Generally, Sparse MAF-AAE achieves SOTA on various scenarios.

![Sparse MAF-AAE main results]()

## Citation
If you use the Sparse MAF-AAE in your research, please cite the following paper:
```
```

## Contact
If you have any question, please contact .

## License 
This project is licensed under the [MIT License](./LICENSE).