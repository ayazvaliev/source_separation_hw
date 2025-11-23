export COMET_API_KEY=0jNECQzQ6wMDAwGQt3Psjj6Zg

HYDRA_FULL_ERROR=1 python3 train.py --config-name=tdanet_train_abl3 data_root=/home/biorn/datasets/ model=tdanet index_dir=indexes dataloader.train.batch_size=27 dataloader.inference.batch_size=27 trainer.save_period=4 trainer.mixed_precision=float32 writer=cometml writer.run_name="tdanet_no_augs_fp32" writer.project_name="report_small_epoch"
