# IceCloudNet
*This is the official repository for the manuscript* ["IceCloudNet: 3D reconstruction of cloud ice from Meteosat SEVIRI"](https://arxiv.org/abs/2410.04135)

Clouds containing ice remain a source of great uncertainty in climate models and future climate projections. IceCloudNet overcomes the limitations of existing satellite observations and fuses the strengths of high spatio-temporal resolution of geostationary satellite data with the high vertical resolution of active satellite retrievals through machine learning. With this work we are providing the research community with a fully temporal and spatial resolved 4D dataset of cloud ice properties enabling novel research ranging from cloud formation and development to the validation of high-resolution weather and climate model simulations.

![](img/icecloudnet_concept.png)

📜 **Cite as**
```bibtex
@article{jeggle2024icecloudnet3d,
      title={IceCloudNet: 3D reconstruction of cloud ice from Meteosat SEVIRI}, 
      author={Kai Jeggle and Mikolaj Czerkawski and Federico Serva and Bertrand Le Saux and David Neubauer and Ulrike Lohmann},
      year={2024},
      eprint={2410.04135},
      archivePrefix={arXiv},
      primaryClass={physics.ao-ph},
      url={https://arxiv.org/abs/2410.04135}, 
}
```

💾 [**Dataset Access at WDC Climate**](https://www.wdc-climate.de/ui/entry?acronym=IceCloudNet_3Drecon_2010)

## Inference 

This repo contains a pretrained model that can be used to create 3D cloud structures from MeteoSat Seviri input.

Steps:
1. Download SEVIRI input data: `src/download_data/MSG_eunmdac_satpy.ipynb`
2. Unzip with `src/unzip_seviri.sh`
3. Create input data for ML model: `src/inference/CreateSeviriWholeAreaTimeSeries.ipynb`
4. Specify `DATA_DIR` variable in `src/inference.py` and set up directory structure as described there
    * mv `helper_files/*  DATA_DIR`
5. Run inference: `run_inference.sh`

![](img/ice_cloud_net_rendering.gif)

## Training

### Download input data

Steps:
1. Download SEVIRI input data: `src/download_data/MSG_eunmdac_satpy.ipynb`
2. Unzip with `src/unzip_seviri.sh`
3. Download DARDAR-Nice from [ICARE data servers](earlier version: https://www.icare.univ-lille.fr/data-access/data-archive-access/?dir=CLOUDSAT/DARDAR-NICE_L2-PRO.v1.00/) → login required. 
    * Currently only DARDAR Nice v1 available on ICARE. Contact the author for v2 (which is used in the paper).

### Preprocess and co-locate SEVIRI and DARDAR-Nice

```bash
python data_preproc.py YYYYMMDD YYYYMMDD n_workers
```

### Run experiment

We are using [Comet ML](https://www.comet.com/) to track experiments. Setup up account and insert your credentials in the necessary places. Setup up environ variables alternatively.

Steps: 
* mv `helper_files/*  /path/to/your/data/directory`
* specify filepaths in `run_experiment.py` and then run 

## Evaluation

Run `EvaluateModel.ipynb`. Requires that co-located patches were created already

## Setup

There are two conda environment files:
* `satpy.yml` used for everything that requires satpy
* `pytorch.yml` used for everything that involves pytorch
