# CV Use-Case Pipeline

This folder now contains only the selected cvusecase flow.

## Files

- `build_cvusecase_dataset.py`: creates `train/valid/test` eye-crop dataset from Blender PNG+JSON pairs.
- `train_cvusecase_model.py`: trains chosen cvusecase model.
- `evaluate_cvusecase_model.py`: evaluates a trained model.
- `run_cvusecase_flow.py`: single entry point (`full`, `train`, `eval`).
- `predict_cvusecase_image.py`: single image prediction helper.
- `requirements_cvusecase.txt`: Python dependencies.

## 1) Build dataset

```powershell
python model_training/cvusecase/build_cvusecase_dataset.py --source-dir <PATH_TO_BLENDER_SOURCE> --out-dir model_training/cvusecase_dataset
```

Input expectations:
- each sample must have matching `<id>.png` and `<id>.json`
- JSON must include `eye_landmarks`, `eye_landmark_mask`, `eye_state_id`

Output layout:

```text
model_training/cvusecase_dataset/
  train/open, train/closed
  valid/open, valid/closed
  test/open,  test/closed
```

## 2) Run cvusecase flow

```powershell
python model_training/cvusecase/run_cvusecase_flow.py --mode full --data model_training/cvusecase_dataset --output model_training/cvusecase/cvusecase_outputs
```

Other modes:

```powershell
python model_training/cvusecase/run_cvusecase_flow.py --mode train --data model_training/cvusecase_dataset --output model_training/cvusecase/cvusecase_outputs
python model_training/cvusecase/run_cvusecase_flow.py --mode eval --data model_training/cvusecase_dataset --output model_training/cvusecase/cvusecase_outputs
```

## 3) Azure end-to-end behavior

The Azure job in [azure/job_cvusecase_full.yaml](azure/job_cvusecase_full.yaml) builds dataset + trains/evaluates in one run.

Input:
- `inputs.blender_source_path`: datastore folder containing matching `<id>.png` and `<id>.json` files.

Internal flow inside the job:
1) `build_cvusecase_dataset.py` reads `inputs.blender_source_path`
2) writes dataset to `${outputs.output_stream}/cvusecase_dataset`
3) `run_cvusecase_flow.py` reads that dataset path for train/eval

Output:
- `outputs.output_stream`: one datastore output folder containing:
  - `cvusecase_dataset/` (train/valid/test)
  - model file
  - metrics JSON
  - report TXT

Submit example:

```powershell
az ml job create --file azure/job_cvusecase_full.yaml --set inputs.blender_source_path=azureml://datastores/workspaceblobstore/paths/<YOUR_BLENDER_SOURCE_FOLDER>/ outputs.output_stream.path=azureml://datastores/workspaceblobstore/paths/<YOUR_OUTPUT_FOLDER>/
```

Notes:
- You do not need to prebuild dataset for this job.
- Output folder path can be a new path; Azure creates it during upload.
