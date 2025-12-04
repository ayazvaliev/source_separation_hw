# Audio Source Separation Model with PyTorch

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Replicate Our Results</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>


## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=3.11

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/3.11.13/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Install `pre-commit` in case you want to contribute:
   ```bash
   pre-commit install
   ```

## How To Replicate Our Results

### Training

1. Firstly, download dataset for training. You can do it with script `load_train_dataset.py`:
```bash
python load_train_dataset.py --output YOUR_SAVE_DIR
```
2. In order to log training process we used [CometML](https://www.comet.com/site/) logger. Set up `COMET_API_KEY` env with your CometML token:
```bash
export COMET_API_KEY="<YOUR COMETML API KEY>"
```
3. Run pre-training of the model:
```bash
python train.py data_root=YOUR_SAVE_DIR writer.run_name="PRETRAIN"
```

4. Then run fine-tuneing of the pre-trained model:
```bash
python train.py --config-name=tdanet_finetune data_root=YOUR_SAVE_DIR trainer.from_pretrained="saved/PRETRAIN/model_best.pth" writer.run_name="FINETUNE"
```

**Important notion**: The model was trained on **1x A100 PCIe** using **PyTorch 2.9.0**, so the training configs and environment were specifically set for this GPU. If you're struggling to run training on another hardware, use `trainer.compile=false` if your GPU doesn't support `torch.compile`, use `dataloader.train/inference.batch_size=` options to reduce batch sizes, or use `requirements_kaggle.txt` if you're struggling to launch training pipeline on **Kaggle** VM.

### Inference and Evaluation
To launch inference and evaluation pipelines from our pre-trained model we suggest to look at `demo.ipynb` Jupyter Notebook, which uses sample dataset of correct structure as an example.

## Credits

1. This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).
2. **TDANet** model was implemented based on the [original TDANet paper](https://arxiv.org/abs/2209.15200)
3. **RTFS-Net** model was implemented based on the [original RTFS-Net paper](https://arxiv.org/abs/2309.17189)

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)



