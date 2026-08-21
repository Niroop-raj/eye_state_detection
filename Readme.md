# eyeCare (cvusecase flow)

This repo is intentionally reduced to one workflow:
- Build eye-crop dataset from Blender PNG+JSON files.
- Train/evaluate only the cvusecase model.
- Run locally or from Azure using one entry script.

No generated datasets, reports, or eval outputs should be committed.

## What is kept

- `model_training/cvusecase/build_cvusecase_dataset.py`
- `model_training/cvusecase/train_cvusecase_model.py`
- `model_training/cvusecase/evaluate_cvusecase_model.py`
- `model_training/cvusecase/run_cvusecase_flow.py`
- `model_training/cvusecase/predict_cvusecase_image.py`
- `azure/job_cvusecase_full.yaml`

## Commands to run right after clone

```powershell
cd eyeCare-main
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r model_training/cvusecase/requirements_cvusecase.txt
```

## Local flow

1) Put your Blender input files in one folder with matching names:
- `<id>.png`
- `<id>.json`
    Here u create a folder and dump all the images in that folder and this is your <PATH_TO_BLENDER_SOURCE>

2) Build dataset:

```powershell
python model_training/cvusecase/build_cvusecase_dataset.py --source-dir <PATH_TO_BLENDER_SOURCE> --out-dir model_training/cvusecase_dataset
```

3) Run train+eval with one script:

```powershell
python model_training/cvusecase/run_cvusecase_flow.py --mode full --data model_training/cvusecase_dataset --output model_training/cvusecase/cvusecase_outputs
```

Optional modes:

```powershell
python model_training/cvusecase/run_cvusecase_flow.py --mode train --data model_training/cvusecase_dataset --output model_training/cvusecase/cvusecase_outputs
python model_training/cvusecase/run_cvusecase_flow.py --mode eval --data model_training/cvusecase_dataset --output model_training/cvusecase/cvusecase_outputs
```

## Azure flow

Azure job file now runs full end-to-end processing in one job:
1) reads Blender source (`.png` + `.json`) from datastore input,
2) builds `train/valid/test` dataset during the run,
3) trains/evaluates model,
4) writes dataset + model + metrics + report to output datastore path.

Azure job input:
- `inputs.blender_source_path`: datastore folder with matching `<id>.png` and `<id>.json` files.

Azure job output:
- `outputs.output_stream`: output folder where all artifacts are uploaded.
- built dataset path in output: `outputs.output_stream/cvusecase_dataset`

Submit:

```powershell
az ml job create --file azure/job_cvusecase_full.yaml --set inputs.blender_source_path=azureml://datastores/workspaceblobstore/paths/<YOUR_BLENDER_SOURCE_FOLDER>/ outputs.output_stream.path=azureml://datastores/workspaceblobstore/paths/<YOUR_OUTPUT_FOLDER>/
```

Important:
- `outputs.output_stream.path` can point to any datastore path you own.
- Azure will create the output folder path if it does not exist.
- The built dataset is saved inside that output location under `cvusecase_dataset`, and the training flow reads it from there in the same run.

## Clean git policy

Do not commit generated artifacts:
- `model_training/cvusecase_dataset/`
- `model_training/cvusecase/cvusecase_outputs/`
- model checkpoints
- report/json result files

`.gitignore` already excludes generated folders.
