# SeedGerm-Vigor

SeedGerm-Vigor is a Streamlit system for automatic seed germination counting, time-series vigor analysis, batch comparison, and experiment report export from petri-dish images.

## Features

- YOLO-based seed detection with germinated / non-germinated counting.
- Single-image analysis for uploaded petri-dish images.
- Local camera capture for quick on-device inspection.
- Time-series germination curve reconstruction.
- Final germination rate, T50, mean germination time, germination speed index, uniformity score, and vigor score.
- Multi-batch comparison for seed lots or treatments.
- Exportable experiment report package with CSV, Markdown, and Word outputs.
- Real evaluation result display from the trained detector.
- OpenCV rule-based baseline for fallback demonstration.

## Model

The demo expects the trained detector at:

```text
models/seed_detector.pt
```

The included weight file is the final YOLO detector exported from the training workflow. Large raw datasets and training workspaces are intentionally excluded from the GitHub package.

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py --server.port 8502
```

Open:

```text
http://127.0.0.1:8502
```

## Input Naming for Multi-Batch Analysis

Use file names that include a batch name and time point:

```text
batchA_00h.jpg
batchA_06h.jpg
batchA_12h.jpg
batchB_00h.jpg
batchB_06h.jpg
batchB_12h.jpg
```

The system automatically groups images by batch and sorts them by time. If no time is found in the file name, it uses the sidebar time interval.

## Project Structure

```text
seed_germination_demo/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── seed_detector.pt
├── results/
│   ├── detection_metrics.csv
│   ├── germination_metrics.csv
│   ├── t50_metrics.csv
│   ├── t50_summary.csv
│   └── test_image_counts.csv
├── scripts/
├── src/
│   ├── detector.py
│   ├── reporting.py
│   ├── vigor.py
│   └── yolo_detector.py
└── data/
    └── sample_sequence/
```

## Public Dataset

Training and evaluation used the public Germination Detection Dataset from Mendeley:

https://data.mendeley.com/datasets/4wkt6thgp6/3

Raw public datasets are not committed to this repository. Download them separately when reproducing training.

## Notes

This system is intended as an agricultural experiment assistance tool. It does not infer seed physiological mechanisms; it provides RGB image-based detection, germination statistics, time-series trend analysis, and report generation.
