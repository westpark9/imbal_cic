# LoCalPFN

This is the codebase pertaining to the paper [Retrieval & Fine-Tuning for In-Context Tabular Models](https://papers.nips.cc/paper_files/paper/2024/hash/c40daf14d7a6469e65116507c21faeb7-Abstract-Conference.html), published in the main conference track of [Advances in Neural Information Processing Systems 37 (NeurIPS 2024)](https://papers.nips.cc/paper_files/paper/2024).

## Preparation
Please download the TabZilla datasets by first cloning the repo
```
git clone https://github.com/naszilla/tabzilla
```
And download the datasets
```
cd tabzilla
python tabzilla_data_preprocessing.py --process_all
```
After that, please clone this repo and softlink the TabZilla dataset folder
```
git clone git@github.com:layer6ai-labs/LoCalPFN.git
cd LoCalPFN
mkdir datasets
ln -s datasets ~/tabzilla
```

## Run Instructions
### TabPFN-kNN
```
python main.py --exp_name="default_knn" knn
```

### LoCalPFN
```
python main.py --exp_name="localpfn" ft
```

# Citation
```
@inproceedings{thomas2024retrieval,
  title={Retrieval \& Fine-Tuning for In-Context Tabular Models},
  author={Thomas, Valentin and Ma, Junwei and Hosseinzadeh, Rasa and Golestan, Keyvan and Yu, Guangwei and Volkovs, Maksims and Caterini, Anthony},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```
