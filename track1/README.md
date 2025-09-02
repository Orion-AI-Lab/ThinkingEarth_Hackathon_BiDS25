# thinkingearth-hackathon-track1

This repo contains resources for track 1 of the [ThinkingEarth Hackathon](https://thinkingearth-hackathon.devpost.com/?_gl=1*u5l5q*_ga*MTU5NDcyMTA0My4xNzUwMDg1MDYw) at [BiDS 25](https://www.bigdatafromspace2025.org/satellite-event-submission), which targets at exploring the usage of Earth observation (EO) foundation models. 

## Tasks
Select or customize one specific topic, then complete one project related to one or more EO foundation models (e.g., [CopernicusFM](https://arxiv.org/abs/2503.11849), [DOFA](https://arxiv.org/abs/2403.15356)). You may explore fine-tuning strategies, adapt models across geographies or modalities, identify limitations, develop novel use cases related to EO applications, or any other interesting ideas. 

Visit the [hackathon page](https://thinkingearth-hackathon.devpost.com/?_gl=1*u5l5q*_ga*MTU5NDcyMTA0My4xNzUwMDg1MDYw) for detailed organization information.

## Submissions
Participants must submit the following materials through the dedicated platform by the end of the development phase:
- Project description outlining the problem addressed, approach taken, and relevance to the selected track
- Source code used to develop the solution (e.g., Jupyter notebooks, scripts, or repositories)
- Documentation explaining how to run the code, key methods used, and any dependencies or requirements
- Demo or presentation materials showcasing the solution and its functionality (e.g., slides, short video, screenshots)

A convenient way is to create a github repo with all the required materials and share the link in your submission:
- a README.md file containing project description, code documentation, and access to the presentations
- source codes
- presentation or link to the presentation

## Judging Criteria
Submissions will be evaluated by a panel of four expert judges based on:
- Effectiveness
- Efficiency
- Novelty
- Clarity
- Impact potential

## Important resources

### Introduction to EO foundation models

In short, EO foundation models are neural network backbones (e.g. ResNet, ViT) pretrained on big EO data (usually in a self-supervised manner without labels), which can then be transferred to various downstream tasks through fine tuning or other task-specific strategies.

See the webinar on September 3rd at 10:00 AM CEST for an official introduction. Slides will be available [here](https://example.com).

### Demo notebooks

Here we prepare some demo jupyter notebooks playing with example EO foundation models.
- Partial tuning (freeze encoder) on classification downstream task: [`track1_demo1_classification.ipynb`](ThinkingEarth_hackathon_track1_demo1_classification.ipynb)
- Partial tuning on segmentation/regression downstream task: [`track1_demo2_segmentation.ipynb`](ThinkingEarth_hackathon_track1_demo2_segmentation.ipynb)
- Simple parameter-efficient fine tuning (LoRA): [`track1_demo3_PEFT.ipynb`](ThinkingEarth_hackathon_track1_demo3_PEFT.ipynb)
- Simple image retrieval: [`track1_demo4_retrieval.ipynb`](ThinkingEarth_hackathon_track1_demo4_retrieval.ipynb)


### Example EO foundation models 

Here we list some EO foundation models as examples, you can pick one/some of them or explore other public models that you are interested in.

| Model name    | Input modality            | Input bands | Architecture | Source code                                                                | Torchgeo support                                                                   |
|---------------|---------------------------|-------------|--------------|----------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Copernicus-FM | Any                       | Any         | ViT          | [GitHub](https://github.com/zhu-xlab/Copernicus-FM/tree/main/Copernicus-FM) | [weights](https://torchgeo.readthedocs.io/en/stable/api/models.html#copernicus-fm)  |
| DOFA          | Any (spectral)            | Any         | ViT          | [GitHub](https://github.com/zhu-xlab/DOFA)                                  | [weights](https://torchgeo.readthedocs.io/en/stable/api/models.html#dofa)           |
| FoMo-Net      | Any (spectral+DEM)        | Any         | ViT          | [GitHub](https://github.com/RolnickLab/FoMo-Bench)                          | -                                                                                  |
| Panopticon    | Any (spectral)            | Any         | ViT          | [GitHub](https://github.com/Panopticon-FM/panopticon)                       | [weights](https://torchgeo.readthedocs.io/en/stable/api/models.html#panopticon)      |
| CROMA         | SAR + Multispectral       | 2 / 12      | ViT          | [GitHub](https://github.com/antofuller/CROMA)                               | [weights](https://torchgeo.readthedocs.io/en/stable/api/models.html#croma)           |
| DeCUR         | SAR / Multispectral       | 2 / 13      | ResNet       | [GitHub](https://github.com/zhu-xlab/DeCUR)                                 | [weights](https://torchgeo.readthedocs.io/en/stable/api/models.html#sentinel-2)      |
| SoftCon       | SAR / Multispectral       | 2 / 13      | ResNet/ViT   | [GitHub](https://github.com/zhu-xlab/softcon)                               | [weights](https://torchgeo.readthedocs.io/en/stable/api/models.html#sentinel-2)      |
| FG-MAE        | SAR / Multispectral       | 2 / 13      | ViT          | [GitHub](https://github.com/zhu-xlab/FGMAE)                                 | [weights](https://torchgeo.readthedocs.io/en/stable/api/models.html#sentinel-2)      |
| Satlas        | RGB / SAR / Multispectral | 3 / 2 / 10  | ResNet/Swin  | [GitHub](https://github.com/allenai/satlas)                                 | [weights](https://torchgeo.readthedocs.io/en/stable/api/models.html#naip)            |
| SSL4EO-S12    | RGB / SAR / Multispectral | 3 / 2 / 13  | ResNet/ViT   | [GitHub](https://github.com/zhu-xlab/SSL4EO-S12)                            | [weights](https://torchgeo.readthedocs.io/en/stable/api/models.html#sentinel-2)      |
| SSL4EO-L      | Multispectral             | 6/7/9/11    | ResNet/ViT   | [GitHub](https://github.com/microsoft/torchgeo)                             | [weights](https://torchgeo.readthedocs.io/en/stable/api/models.html#landsat)         |
| SatMAE        | RGB / Multispectral       | 3 / Any     | ViT          | [GitHub](https://github.com/sustainlab-group/SatMAE)                        | -                                                                                  |
| Scale-MAE     | RGB                       | 3           | ViT          | [GitHub](https://github.com/bair-climate-initiative/scale-mae)              | [weights](https://torchgeo.readthedocs.io/en/stable/api/models.html#sensor-agnostic) |
| SpectralEarth     | Hyperspectral                       | 202           | Resnet/ViT          | [GitHub](https://github.com/AABNassim/spectral_earth)              | - |
| ... | ... | ... | ... | ... | ... |

### Example downstream tasks

Here we list some EO downstream tasks/datasets from [Copernicus-Bench](https://github.com/zhu-xlab/Copernicus-FM/tree/main/Copernicus-Bench) (also available in [TorchGeo](https://torchgeo.readthedocs.io/en/stable/api/datasets.html#copernicus-bench)) as examples, you can pick one/some of them or explore other specific tasks that you are interested in.

| Name           | Modality      | Task                                | # Images        | Image Size         | # Classes |
|----------------|---------------|-------------------------------------|-----------------|--------------------|-----------|
| Cloud-S2       | Multispectral | segmentation (cloud)                | 1699/567/551    | 512x512x13         | 4         |
| Cloud-S3       | Multispectral | segmentation (cloud)                | 1197/399/399    | 256x256x21         | 5         |
| EuroSAT-S1     | SAR           | classification (LULC)               | 16200/5400/5400 | 64x64x2            | 10        |
| EuroSAT-S2     | Multispectral | classification (LULC)               | 16200/5400/5400 | 64x64x13           | 10        |
| BigEarthNet-S1 | SAR           | classification (LULC)               | 11894/6117/5991 | 120x120x12         | 19        |
| BigEarthNet-S2 | Multispectral | classification (LULC)               | 11894/6117/5991 | 120x120x12         | 19        |
| LC100Cls-S3    | Multispectral | classification (LULC)               | 5181/1727/1727* | 96x96x21           | 23        |
| DFC2020-S1     | SAR           | segmentation (LULC)                 | 3156/986/986    | 256x256x13         | 10        |
| DFC2020-S2     | Multispectral | segmentation (LULC)                 | 3156/986/986    | 256x256x13         | 10        |
| LC100Seg-S3    | Multispectral | segmentation (LULC)                 | 5181/1727/1727* | 96x96x21 (288x288) | 23        |
| Flood-S1       | SAR           | change detection (flood)            | 3000/1000/1000  | 224x224x2          | 3         |
| LCZ-S2         | Multispectral | classification (local climate zone) | 15000/5000/5000 | 32x32x10           | 17        |
| Biomass-S3     | Multispectral | regression (biomass)                | 3000/1000/1000* | 96x96x21 (288x288) | 1         |
| AQ-NO2-S5P     | Air pollutant | regression (air quality)            | 1480/493/494*   | 56x56x1            | 1         |
| AQ-O3-S5P      | Air pollutant | regression (air quality)            | 1480/493/494*   | 56x56x1            | 1         |

Other useful resources:
- [TorchGeo datasets](https://torchgeo.readthedocs.io/en/stable/api/datasets.html#) 
- [EarthNets datasets](https://earthnets.retool.com/embedded/public/676aa812-0dca-4e3b-a596-b043d852571d)
- [Geo-Bench](https://github.com/ServiceNow/geo-bench)
- [PANGAEA](https://github.com/VMarsocci/pangaea-bench)
- ...

### Example topics

You can pick or combine one or more example topics below, as well as explore other topics you find interesting and valuable.

- Do foundation models provide more benefits on some tasks / task types while less on some others?
- Do different foundation models have different strengths and weaknesses?
- Does it hold that larger foundation models perform better than smaller ones?
- Can one foundation model trained on one modality be transferred to another modality?
- How do foundation models perform across different amount of labels (many to zero)?
- How do foundation models perform when labels are noisy?
- How do foundation models perform across different geographical regions?
- How do foundation models perform across different time periods?
- How do foundation models perform across different spatial resolutions?
- Explore parameter efficient fine tuning techniques (LoRA, bias, Prompt, and more?)
- Explore what EO FMs "see" in embeddings through feature visualization
- Explore explanability methods to interpret what a foundation model focus on
- Explore whether embeddings preserve semantic similarity across tasks
- Explore novel real world tasks beyond benchmark suites
- Explore combining foundation models with other data sources (e.g. socio-economic data) for complex problems
- ......

