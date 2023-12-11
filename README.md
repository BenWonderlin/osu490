# osu490

A ML performance point algorithm. Completed as part of CPSC 490: Senior Project.

#### [Project Proposal](https://docs.google.com/document/d/16lTe_-yVOx2Tm5fqtjL-9jL9Q8OWa6cEd3D-b38LDrw/edit?usp=sharing)

#### [Project Proposal Presentation](https://docs.google.com/presentation/d/1GiD7Vj2NPH_H91t13GZrv2OhZDkSl3OUOmD6JZb9Pyw/edit?usp=sharing)

#### [Midterm Presentation](https://docs.google.com/presentation/d/1cMgcon8YuvL-X5EikRF4m6VSf-QQZ_8uVAs-Pfb0T_w/edit?usp=sharing)

#### [Final Poster](https://drive.google.com/file/d/15RLx-jx8CTsCsaT0eFim_OsPXikoztkE/view?usp=sharing)

#### [Final Report](https://docs.google.com/document/d/1dxcpPlXla5SVATwR15Pf8kRxHcr1xYhVloNXdglCNVQ/edit?usp=sharing)



<br>

To calculate difficulty estimates, use `evaluate_difficulty.py` as follows:

```
python evaluate_difficulty.py --replay_dir REPLAY_DIR --beatmap_dir BEATMAP_DIR [--model MODEL]
```

where the replay directory contains `.osr` files being evaluated, the beatmap directory contains the corresponding `.osu` files, and the model is a string indicating the model type. Note that the beatmap files must be named by their md5 hashes like those in the [o!rdr replay dataset](https://www.kaggle.com/datasets/skihikingkevin/ordr-replay-dump). The model type must be `"naive"`, `"seq"`, or `"seq2seq"`. Use the `help` flag for more information.

For the sake of consistency, it is strongly recommended to use the conda environments specified in [requirements_cpu.txt](requirements_cpu.txt) and [requirements_gpu.txt](requirements_gpu.txt).
