# sign-to-subtitle

Official PyTorch implementation for the following paper:
> **Deep Understanding of Sign Language for Sign to Subtitle Alignment**<br>
> [Youngjoon Jang](https://art-jang.github.io)\*, [Jeongsoo Choi](https://choijeongsoo.github.io)\*, [Junseok Ahn](https://junseok520.github.io), [Joon Son Chung](https://mm.kaist.ac.kr/joon)<br>
> IEEE Transactions on Multimedia<br>
> \[[Paper](https://arxiv.org/abs/2503.03287)\]

## Model Checkpoints
Download and place this checkpoint as `inference_output/finetune_self_training/checkpoints/model_best.pt`

  Dataset | frame-acc (%) ↑ | F1@.10 ↑ | F1@.25 ↑ | F1@.50 ↑ | Model
  :---:|:---:|:---:|:---:|:---:|:---:
  [BOBSL](https://www.robots.ox.ac.uk/~vgg/data/bobsl) | 77.22 | 81.39 | 75.03 | 63.81 | [download](https://drive.google.com/file/d/1TcloBzQt2TQkJ9zNxHE7js0G_Qpq6HXh/view)

## Train
```bash
# pretrain with word data
commands/word_pretrain.sh
# train with audio-aligned subtitles
commands/train.sh
# finetune with munally aligned subtitles
commands/finetune.sh
# generate pseudo labels for self-training
commands/test_for_self_training.sh
# further train with pseudo labelled subtitles
commands/train_self_training.sh
# final finetune with munally aligned subtitles
commands/finetune_self_training.sh
```

## Test
```bash
commands/test_self_training.sh
```

## Acknowledgement
This repository is built using [subtitle_align](https://github.com/hannahbull/subtitle_align). We appreciate the open source of the project.

## Citation
If our work is useful for you, please cite the following paper:
```bibtex
@article{jang2025deep,
  title={Deep Understanding of Sign Language for Sign to Subtitle Alignment},
  author={Jang, Youngjoon and Choi, Jeongsoo and Ahn, Junseok and Chung, Joon Son},
  journal={arXiv preprint arXiv:2503.03287},
  year={2025}
}
```

