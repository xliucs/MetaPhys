# MetaPhys
Paper Title: MetaPhys: Few-Shot Adaptation for Non-Contact Physiological Measurement (ACM CHIL-2021)


# Sample Waveforms 

## UBFC 

![Alt Text](./UBFC.gif)

![Alt Text](./ubfc_sample_waveforms.jpg)

## MMSE 

![Alt Text](./MMSE.gif)

![Alt Text](./mmse_sample_waveforms.jpg)

# Code 

`train_higher.py` is the training script of our proposed MetaPhys. 

`train_txt.py` is the training script for regular supervised training. 

`train_txt_ft.py` is the training script for supervised training + fine-tuning. 

`higher_model.py` is the implemnetation of TS-CAN. 

`rppg_dataset.py` is the dataloader for MetaPhys. 

`data_generator.py` is the dataloader for regular supervised training. 

`splitter.py` is the code for spliting support and query sets. 



# Acknowledgement 

MetaPhys's dataloading schema is based on pytorch-meta (https://github.com/tristandeleu/pytorch-meta), and our inner/outer loop optimizaton is based on higher (https://github.com/facebookresearch/higher). 

