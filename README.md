# Master Thesis: Text2Gloss2Pose2Sign: Gloss- and Pose-guided Sign Language Production with LLMs

This repository contains all source code developed for the master thesis "Text2Gloss2Pose2Sign: Gloss- and Pose-guided Sign Language Production with LLMs", including data collection, preprocessing, augmentation, analysis, evaluation, training scripts, pod configurations for communication and execution on the BHT cluster, and more.

* Author: Frederik Busch
* BHT Student ID: 925110
* E-mail: frederik.bln98@gmail.com
* Submission Date: August 31, 2025
* Berliner Hochschule für Technik (BHT)

## Getting Started

### 0. Initial Setup
* Use Python Version 3.12.2
* Install dependencies: `conda env create -f environment.yaml` - There may be some unnecessary dependencies included, as I used the same Conda environment for multiple projects during development and unfortunately lost track of which packages belong to which project. I didn’t have time to create a clean, dedicated environment. Feel free to delete the environment afterward to avoid cluttering your storage with unrelated packages from my other work.
* Actiavte the conda environment `conda activate cv`.

The following scripts need to be executed in order to get the desired results and to be able to use the SLP pipelines functionality in the end.

*Important Note:* To gather and preprocess all data you need about 400 GB of free storage. I suggest using some kind of hard drive. In most scripts the `base directory` where everything is happening is set to `/Volumes/IISY/DGSKorpus/` this needs to be adapated according to your used storage location!

### 1. Data Collection
Under `src/data-processing/scraping`...
1. there is the `dgs_webscraper.py` which scrapes all raw data from DGS-Korpus Release 3 web pages.
2. there is the `dgs_types_webscraper.py` which scrapes all types defined in the DGS-Korpus Release 3 web pages. Types are synonym with glosses.

### 2. Data Preprocessing
Under `src/data-processing/`...
1. there is `filter_transcripts.py`. This applies initial filtering of the DGS-Korpus transcripts.
2. there is `split_transcripts_by_speaker.py`. Further preprocessing and splitting of the transcripts into speaker A and speaker B.
3. `exclude_problematic_glosses.py`. This filters the transcripts further and excludes problematic glosses.
4. `text2gloss_mapper.py`. Maps full sentences to their corresponding gloss sequences, creating the training data for the Text2Gloss LLM.
5. `gloss2pose_mapper.py`. Maps glosses to their corresponding frames in the OpenPose json files.
6. `create_gloss2pose_dictionary.py`. Creates the final 1:1 lookup Gloss2Pose dictionary used later for the Gloss2Pose mdoule of the SLP pipeline. The resulting `gloss2pose_dictionary.json` needs to be saved under `src/pipeline/resources/` for later use!
7. (OPTIONAL) `sentence2pose_mapper.py`. Additional preprocessing script that creates mappings between entire sentences, their corresponding gloss sequence and the correpsonding pose sequences from the OpenPose json files. This was implemented to have data for the Sign-Language Back-Translation which was not evaluate in the end. The resulting data could also be used for training a data-driven, generative Gloss2Pose model in the future.
8. (OPTIONAL) `sentence2vide_mapper.py`. Additional preprocessing script that creates mappings between entire sentences, their corresponding gloss sequence and the correpsonding timestamps in the video files from the DGS-Korpus Release 3. This was implemented to have data for the Sign-Language Back-Translation which was not evaluate in the end.

### 3. Data Augmentation (only needed if you really want to train the model again)
1. Under `src/data-processing/`there is the `back_translation.py` script which implements data augmentation of the Text2Gloss data resulting from the previous data preprocessing step 4. This takes a long time to finish and is only needed if you want to train the Text2Gloss model on the final augmented Text2Gloss data.

### 4. BHT Cluster: Training
Under `bht-cluster/` there are all the pod configurations used to communicate and use the cluster for Training and Finetuning purposes. There you can also find some explaining `.txt` files which explain how the pods were used.

Under `bht-cluster/deepseek-finetuning/` there is the `train_deepseek_distill.py` script which was used to finetune the Text2Gloss LLM. Here you can also find the `text2gloss.py` script which was deployed on the cluster to start inference of the finetuned Text2Gloss LLM to translate German sentences into gloss sequences for the initial part of the SLP pipeline.

### 5. Using the Gloss2Pose module
1. Under `src/pipeline/gloss-similarity` you need to execute `get_unique_glosses_from_dictionary.py` and then `build_gloss_embeddings.py` for the Gloss Matcher at the beginning of the Gloss2Pose module to work correctly.
2. The resulting files: `gloss_embeddings.npz` and `gloss_to_idx.json` need to be placed under `src/pipeline/resources/` alongside the Gloss2Pose dictionary.

Usage Example:

`python gloss2pose.py --glosses 'BEREICH1A,INTERESSE1A,MERKWÜRDIG1,GEBÄRDEN1A,FASZINIEREND2,GEBÄRDEN1A,SPIELEN2,BEREICH1A,INTERESSE1A,SPIELEN2' --output-filename example-video --config-filename example-config.yml`

`python gloss2pose.py -g 'BEREICH1A,INTERESSE1A,MERKWÜRDIG1,GEBÄRDEN1A,FASZINIEREND2,GEBÄRDEN1A,SPIELEN2,BEREICH1A,INTERESSE1A,SPIELEN2' -o example-video -c example-config.yml`

Results will be saved under `pipeline/outputs/pose-sequence-videos`.

### Additional scripts:
Under `src/data-processing/evaluation` you can find the scripts that calculate the gloss time metrics and various scripts that were implemented to visualize and plot some of the data for figures used in the thesis.

Under `src/mimicmotion/` you can find the `inference.py` script that was adapted for the SLP pipeline proposed in this thesis. If you want to run the entrie MimicMotion pipeline you need at least an NVIDIA A100 and should first follow their [installation guide](https://github.com/Tencent/MimicMotion?tab=readme-ov-file#quickstart) and finally replace their `inference.py` with the one provided here!

Under `src/more/` you can find various utility scripts that have been implemented at some point during the development but which can not be clearly grouped to any specific purpose so they are just dumped here. Feel free to check them out as well.

### Rest of the SLP pipeline
As previously mentioned, using the Text2Gloss LLM and the Pose2Sign module requires access to high-end GPUs, such as those available on the BHT Data Science Cluster. Training the Text2Gloss model is a time-consuming process due to its large size, and unfortunately, I was unable to upload the pretrained model to the Git repository. Additionally, running inference with both the Text2Gloss and Pose2Sign models also demands significant GPU resources. I hope it's understandable that these features cannot be easily tested without the appropriate hardware.


## Project Structure Tree
```text
.
└── text2gloss2pose2sign/
    ├── bht-cluster/
    │   ├── deepseek-finetuning/
    │   │   ├── data-augmentation
    │   │   ├── k-cross-validation
    │   │   ├── phoenix-weather
    │   │   ├── text2gloss_data
    │   │   └── gloss2text
    │   ├── mBART
    │   └── mimicmotion/
    │       ├── docker
    │       └── human-eval-gen
    ├── src/
    │   ├── data-processing/
    │   │   ├── evaluation
    │   │   ├── problematic-glosses
    │   │   └── scraping
    │   ├── mimicmotion
    │   ├── pipeline/
    │   │   └── gloss-similarity
    │   └── visualization
    └── more
```
