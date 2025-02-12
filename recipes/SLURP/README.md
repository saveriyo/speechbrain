# SLU recipes for SLURP
This folder contains recipes for spoken language understanding (SLU) with [SLURP](https://zenodo.org/record/4274930#.YEFCYHVKg5k).

## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
pip install -r extra_requirements.txt
```

### Direct recipe
The "direct" maps the input speech directly to semantics using a seq2seq model.

```
cd direct
python train.py hparams/train.yaml

# for wav2vec2/HuBert
python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml
```

### Tokenizer recipe
(You don't need to run this because the direct recipe downloads a tokenizer, but you can if you'd like to train a new tokenizer for SLURP.)

Run this to train the tokenizer:

```
cd Tokenizer
python train.py hparams/tokenizer_bpe51.yaml
```

### NLU recipe
The "NLU" recipe takes the true transcript as input rather than speech and trains a seq2seq model to map the transcript to the semantics.

```
cd NLU
python train.py hparams/train.yaml
```


# Performance summary
Note: SLURP comes with a tool for measuring SLU-F1 and other metrics.
The recipes here dump the model outputs to a file called `predictions.jsonl` in the `results` folder.
You can compute the metrics by feeding `predictions.jsonl` into the [SLURP evaluation tool](https://github.com/pswietojanski/slurp/tree/master/scripts/evaluation).

The following results were obtained on a 48 GB RTX 8000 (the recipe has also been successfully tested on a 12 GB Tesla K80):

| Model	| scenario (accuracy) | action (accuracy) | intent (accuracy) | Word-F1 | Char-F1 | SLU-F1 | Training time | Model link |
|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Direct | 81.73 | 77.11 | 75.05 | 61.24 | 65.42 | 63.26 | 1 hour per epoch | https://www.dropbox.com/scl/fo/c0rm2ja8oxus8q27om8ve/h?rlkey=irxzl1ea8g7e6ipk0vuc288zh&dl=0 |
| Direct (HuBert) | 91.24 | 88.47 | 87.54 | 72.93 | 77.40 | 75.10 | 4 hours per epoch | https://www.dropbox.com/scl/fo/c0rm2ja8oxus8q27om8ve/h?rlkey=irxzl1ea8g7e6ipk0vuc288zh&dl=0 |

| Model	| scenario (accuracy) | action (accuracy) | intent (accuracy) | Training time |
|:---:|:-----:|:-----:|:-----:|:-----:|
| NLU | 90.81 | 88.29 | 87.28 | 40 min per epoch |

# Inference with easy-to-use Interface
The Direct (HuBert) model can be found on HuggingFace (coupled with an easy-inference interface):

https://huggingface.co/speechbrain/SLU-direct-SLURP-hubert-enc

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{ravanelli2024opensourceconversationalaispeechbrain,
      title={Open-Source Conversational AI with SpeechBrain 1.0},
      author={Mirco Ravanelli and Titouan Parcollet and Adel Moumen and Sylvain de Langen and Cem Subakan and Peter Plantinga and Yingzhi Wang and Pooneh Mousavi and Luca Della Libera and Artem Ploujnikov and Francesco Paissan and Davide Borra and Salah Zaiem and Zeyu Zhao and Shucong Zhang and Georgios Karakasidis and Sung-Lin Yeh and Pierre Champion and Aku Rouhe and Rudolf Braun and Florian Mai and Juan Zuluaga-Gomez and Seyed Mahed Mousavi and Andreas Nautsch and Xuechen Liu and Sangeet Sagar and Jarod Duret and Salima Mdhaffar and Gaelle Laperriere and Mickael Rouvier and Renato De Mori and Yannick Esteve},
      year={2024},
      eprint={2407.00463},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.00463},
}
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```


