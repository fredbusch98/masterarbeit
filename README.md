# Master Thesis: Text2Gloss2Pose2Sign: Gloss- and Pose-guided Sign Language Production with LLMs

This repository contains all source code developed for the master thesis "Text2Gloss2Pose2Sign: Gloss- and Pose-guided Sign Language Production with LLMs", including data collection, preprocessing, augmentation, analysis, evaluation, training scripts, pod configurations for communication and execution on the BHT cluster, and more.

* Author: Frederik Busch
* BHT Student ID: 925110
* E-mail: frederik.bln98@gmail.com
* Submission Date: August 31, 2025
* Berliner Hochschule für Technik (BHT)

## Project Structure Tree

```text
.
└── text2gloss2pose2sign/
    ├── bht-cluster/
    │   ├── deepseek-finetuning/
    │   │   ├── data-augmentation
    │   │   ├── k-cross-validation
    │   │   ├── phoenix-weather
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
    │   │   └── setup
    │   └── visualization
    └── more
```

## Getting Started

### 0. Initial Setup
* Use Python Version `3.12.2`
* Install dependencies: 
```sh
conda env create -f environment.yaml
``` 
**Important Note:** There may be some unnecessary dependencies included, as I used the same Conda environment for multiple projects during development and unfortunately lost track of which packages belong to which project. I didn’t have time to create a clean, dedicated environment. Feel free to delete the environment afterward to avoid cluttering your storage with unrelated packages from my other work.
* Actiavte the conda environment:

```sh
conda activate cv`
```

The following scripts need to be executed in order to get the desired results and to be able to use the SLP pipelines functionality in the end.

*Important Note:* Gathering and preprocessing all required data will need about 400 GB of free storage. I recommend using a dedicated hard drive. Wherever the file path `/Volumes/IISY/DGSKorpus` (hard drive used for DGS-Korpus data storage) appears in the scripts, you must adapt it to match your own file system, folder structure and selected storage option for everything to work correctly.

### 1. Data Collection
Under `src/data-processing/scraping`...
1. there is the `dgs_webscraper.py` which scrapes all raw data from DGS-Korpus Release 3 web pages:

```sh
python dgs_webscraper.py
```

2. there is the `dgs_types_webscraper.py` which scrapes all types defined in the DGS-Korpus Release 3 web pages. Types are synonym with glosses:

```sh
python dgs_types_webscraper.py
```

### 2. Data Preprocessing
Under `src/data-processing/`...
1. there is `filter_transcripts.py`. This applies initial filtering of the DGS-Korpus transcripts:

```sh
python filter_transcripts.py
```

2. there is `split_transcripts_by_speaker.py`. Further preprocessing and splitting of the transcripts into speaker A and speaker B:

```sh
python split_transcripts_by_speaker.py
```

3. `exclude_problematic_glosses.py`. This filters the transcripts further and excludes problematic glosses:

```sh
python exclude_problematic_glosses.py
```

4. `text2gloss_mapper.py`. Maps full sentences to their corresponding gloss sequences, creating the training data for the Text2Gloss LLM:

```sh
python text2gloss_mapper.py
```

5. `gloss2pose_mapper.py`. Maps glosses to their corresponding frames in the OpenPose json files:

```sh
python gloss2pose_mapper.py
```

6. `create_gloss2pose_dictionary.py`. Creates the final 1:1 lookup Gloss2Pose dictionary used later for the Gloss2Pose mdoule of the SLP pipeline. The resulting `gloss2pose_dictionary.json` should have been saved under `src/pipeline/resources/` for later use!

```sh
python create_gloss2pose_dictionary.py
```

7. Under `src/data-processing/evaluation` run `calculate_gloss_time_metrics.py` followed by `evaluate_gloss_time_metrics.py`. These scripts calculate and evaluate the Gloss Time Metrics described in the thesis and the results of the scripts will be used in the Gloss2Pose module of the pipeline for smooth frame interpolation.

```sh
python calculate_gloss_time_metrics.py
```

```sh
python evaluate_gloss_time_metrics.py
```

8. (OPTIONAL) `sentence2pose_mapper.py`. Additional preprocessing script that creates mappings between entire sentences, their corresponding gloss sequence and the correpsonding pose sequences from the OpenPose json files. This was implemented to have data for the Sign-Language Back-Translation which was not evaluate in the end. The resulting data could also be used for training a data-driven, generative Gloss2Pose model in the future.
9. (OPTIONAL) `sentence2vide_mapper.py`. Additional preprocessing script that creates mappings between entire sentences, their corresponding gloss sequence and the correpsonding timestamps in the video files from the DGS-Korpus Release 3. This was implemented to have data for the Sign-Language Back-Translation which was not evaluate in the end.

### 3. Data Augmentation (only needed if you really want to train the model again)
1. Under `src/data-processing/`there is the `back_translation.py` script which implements data augmentation of the Text2Gloss data resulting from the previous data preprocessing step 4. This takes a long time to finish and is only needed if you want to train the Text2Gloss model on the final augmented Text2Gloss data.

### 4. BHT Cluster: Training
Under `bht-cluster/` there are all the pod configurations used to communicate and use the cluster for Training and Finetuning purposes. There you can also find some explaining `.txt` files which explain how the pods were used.

Under `bht-cluster/deepseek-finetuning/` there is the `train_deepseek_distill.py` script which was used to finetune the Text2Gloss LLM. Here you can also find the `text2gloss.py` script which was deployed on the cluster to start inference of the finetuned Text2Gloss LLM to translate German sentences into gloss sequences for the initial part of the SLP pipeline.

### 5. Using the Gloss2Pose module
0. If you want to skip the data collection and preprocessing steps but still test the Gloss2Pose model, you can simply download the [`gloss2pose_dictionary.json`](https://drive.google.com/file/d/1rd0UQKMvWefWksdAw8MifO6U9qEM5fZD/view?usp=sharing) from Google Drive and save it under `src/pipeline/resources/`! Also download [`evaluated_gloss_time_metrics_filtered.csv`](https://drive.google.com/file/d/1qP9mZyNJlpuUjV2IuOsCBPXOx7gKX-J9/view?usp=sharing) from the Google Drive and save it under `src/pipeline/resources/`!

1. Under `src/pipeline/setup` you need to first execute the following script:
```sh
python get_unique_glosses_from_dictionary.py
```
2. Then run the following script in the same directory to build the gloss embeddings needed for the Gloss Matche of the Gloss2Pose module to work correctly:
```sh
python build_gloss_embeddings.py
```
3. Finally you also need to run the following script (also under `src/pipeline/`):
```sh
python get_gloss_times_for_frame_interpolation.py
```
**Important Note:** Previously you must download [`evaluated_gloss_time_metrics_filtered.csv`](https://drive.google.com/file/d/1qP9mZyNJlpuUjV2IuOsCBPXOx7gKX-J9/view?usp=sharing) from the Google Drive and save it under `src/pipeline/resources/` if you haven't done all the data collection and preprocessing steps described above which would create this file!

**Gloss2Pose Module Usage Examples:**

```sh
python gloss2pose.py --glosses 'BEREICH1A,INTERESSE1A,MERKWÜRDIG1,GEBÄRDEN1A,FASZINIEREND2,GEBÄRDEN1A,SPIELEN2,BEREICH1A,INTERESSE1A,SPIELEN2' --output-filename example-video --config-filename example-config.yml
```

```sh
python gloss2pose.py -g 'BEREICH1A,INTERESSE1A,MERKWÜRDIG1,GEBÄRDEN1A,FASZINIEREND2,GEBÄRDEN1A,SPIELEN2,BEREICH1A,INTERESSE1A,SPIELEN2' -o example-video -c example-config.yml
```

Results will be saved under `src/pipeline/outputs/pose-sequence-videos`.

### Rest of the SLP pipeline
As previously mentioned, using the Text2Gloss LLM and the Pose2Sign module requires access to high-end GPUs, such as those available on the BHT Data Science Cluster. Training the Text2Gloss model is a time-consuming process due to its large size, and unfortunately, I was unable to upload the pretrained model to the Git repository. Additionally, running inference with both the Text2Gloss and Pose2Sign models also demands significant GPU resources. I hope it's understandable that these features cannot be easily tested without the appropriate hardware.

### Text2Gloss Data
If you want to see the Text2Gloss datasets that were extracted, created and augmented without going through the entire data collection and preprocessing procedure, you can simply download them [here](https://drive.google.com/file/d/1g76BcB9G071Mh6nQa2wuDnc-JRSHKpnj/view?usp=sharing).

### Pose Sequence Smoothness Evaluation
If you want to use the script `src/data-processing/evaluation/pose_sequence_smoothness_evaluation.py` you have to download the pose sequence data that was generated during the Human Evaluation from [here](https://drive.google.com/file/d/1W8Z7ODe_GldW_cH95_kysVin2m6CoUzR/view?usp=sharing) and save the extracted folder in the `src/data-processing/evaluation/` directory before running the script.
```sh
python pose_sequence_smoothness_evaluation.py
```

### Additional scripts:
Under `src/data-processing/evaluation` you can find the scripts that calculate the gloss time metrics and various scripts that were implemented to visualize and plot some of the data for figures used in the thesis.

Under `src/mimicmotion/` you can find the `inference.py` script that was adapted for the SLP pipeline proposed in this thesis. If you want to run the entire MimicMotion pipeline you need at least an NVIDIA V100 and should first follow their [installation guide](https://github.com/Tencent/MimicMotion?tab=readme-ov-file#quickstart) and finally replace their `inference.py` with the one provided here! Additionally you need to replace their `environment.yaml` with one provided under `src/mimicmotion/` since their original one had some flaws and needed to be updated to be running correctly with the setup on the BHT cluster. Finally check the `how-to-run-setup.txt` under `bht-cluster/deepseek-finetuning/` for further instructions.

Under `src/more/` you can find various utility scripts that have been implemented at some point during the development but which can not be clearly grouped to any specific purpose so they are just dumped here. Feel free to check them out as well.

## End-to-End Workflow of the Text2Gloss2Pose2Sign SLP Pipeline

### 1. Text → Gloss
1. In the `bht-cluster/deepseek-finetuning/` directory, run:
   ```bash
   ./setup.sh
   ```
   Wait until it completes successfully.

2. Start bash in the pod:
   ```bash
   kubectl -n [your_namespace] exec -it deepseek-finetune -- bash
   ```

3. Navigate to the correct directory:
   ```bash
   cd /storage/text2gloss-finetune/
   ```

4. Generate gloss sequence with finetuned DeepSeek:
   ```bash
   python text2gloss.py "Ein deutscher Satz."
   ```
   This runs inference with the finetuned DeepSeek model and prints the generated gloss sequence.

5. **Copy** the generated gloss sequence from the console output.

---

### 2. Gloss → Pose
6. In the `src/pipeline/` directory, run:
   ```bash
   python gloss2pose.py -g "INSERT GLOSS SEQUENCE" -o example-video -c example-config.yml
   ```
   Follow the console log instructions to finally generate the realistic video with MimicMotion.

---

### 3. Pose → Sign Video
7. In the `bht-cluster/mimicmotion/` directory, run:
   ```bash
   ./setup.sh
   ```
   - Generate your own HuggingFace token for the required diffusion model used by MimicMotion.
   - Add the token to a `.env` file.

8. Navigate to the correct directory:
   ```bash
   cd /storage/MimicMotion/
   ```

9. Start inference:
   ```bash
   python inference.py configs/example-config.yml
   ```

10. After successful inference, follow the console log instructions to **copy the generated sign video from the BHT cluster to your local device**.

**Important Note:** All Kubernetes pods and the associated PVC (PersistentVolumeClaim) must be created within your personal namespace on the BHT cluster. Any scripts intended for execution on the cluster must be uploaded to this PVC volume. Additionally, the MimicMotion project must be cloned from its official GitHub repository directly into the PVC volume. Once cloned, replace the `environment.yaml` and `inference.py` files with the custom versions adapted for the _Text2Gloss2Pose2Sign_ pipeline as described in this thesis. Both replacement files are located under `src/mimicmotion/`.

