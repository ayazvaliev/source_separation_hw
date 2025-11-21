export COMET_API_KEY=""

HYDRA_FULL_ERROR=1 python3 train.py --config-name=tdanet_train data_root=/home/biorn/datasets/ model=tdanet index_dir=indexes transforms=transforms_no_augs dataloader.train.batch_size=15 dataloader.inference.batch_size=15 trainer.compile=false trainer.save_period=10 trainer.val_step=10 trainer.n_epochs=50 trainer.mixed_precision=float32 writer=cometml writer.run_name="tdanet_no_augs_fp32"
