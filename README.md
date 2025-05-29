## Towards Structure-aware Model for Multi-modal Knowledge Graph Completion


Source code of the paper "Towards Structure-aware Model for Multi-modal Knowledge Graph Completion".This paper was accepted for TMM'2025.






## Requirements
* python>=3.9
* torch>=2.0 
* transformers
* scipy
* tqdm
...

All experiments are run with 8 V100(32GB) GPUs.

## How to run simply
For better reproducibility of the paper, we provide a simple one-stop operation to run the TSAM model.





For DB15K and MKG-W  datasets, we use files from [DB15K](https://github.com/mniepert/mmkb) and  [MKG-W](https://github.com/quqxui/MMRNS).

### MKG-W dataset

Step 1, Please download the [tokens folder](https://drive.google.com/file/d/1lFVEIe5_G_dw_K2wzvnnWKYYp2hEPWG2/view?usp=sharing
) from Google drive and put it in the TSAM folder. (Due to GitHub storage restrictions, we have stored all processed tokens information in Google drive)


Step 2, Install the model and pre-install related environment
```
pip install -r requirements.txt
```

Step 3, Training and evaluate the model
```
bash train_MKG_W.bash
```



Feel free to change the output directory to any path you think appropriate.




# further 
1.If you need it, we also provide the ckpt from our models in the [ckpt](https://drive.google.com/file/d/1WTj2iotw0NMtXDoUOBZ8_s150JswUM2K/view?usp=sharing) directory.

2.You can download various transformer-based models from [HuggingFace](https://huggingface.co/) on your own and conduct your own experiments based on the "save_token_embedding.py" py scripts.





