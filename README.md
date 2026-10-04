# Prototypical Networks for Early Knee Osteoarthritis Grading

Code, result tables and figures for the paper

> **Prototypical Learning for Early Knee Osteoarthritis Detection: Class-Imbalanced Grading and Label-Efficient Adaptation Under Domain Shift**

The study compares a conventional softmax CNN with prototypical networks for
five-grade Kellgren-Lawrence (KL) grading of knee radiographs, with a focus on
the early grade (KL I). It also tests the trained models on an independent
dataset from a different X-ray device.

## Summary of results

- Internal test set (1,656 images, patient-level split, mean of three seeds):
  prototypical networks raise KL I F1 from 0.29-0.31 (softmax) to 0.36-0.41 and
  increase quadratic weighted kappa (QWK). Overall accuracy is not improved
  (0.65-0.69).
- Most of the gain comes from the class-balanced nearest-prototype decision
  rule. The ordinal loss and the CBAM attention module give no consistent
  additional benefit.
- External set (1,292 images): all models degrade without adaptation
  (macro-F1 0.25-0.34). With a few labelled images per grade, prototypes built
  from prototypical embeddings adapt best at K = 5 and keep the highest QWK at
  every K; at K = 20 a linear probe on softmax features is as good or better
  on macro-F1.

All numbers in the paper are in `tables/` and `external/`.

## Repository contents

| Path | Content |
|---|---|
| `01_train.ipynb` | Trains one model (one model type, one backbone, one seed) and evaluates it on the test set |
| `02_analysis.ipynb` | Reads the 30 trained runs and produces the tables and figures of the internal evaluation |
| `03_external_validation.ipynb` | Prepares the external dataset and runs the zero-shot, K-shot and linear-probe experiments |
| `tables/` | CSV files behind Tables 1 and 3-9 of the paper |
| `figures/` | Figures 5-9 of the paper (PNG at 400 dpi and PDF) |
| `external/` | CSV files and figures of the external validation (Tables 10-11, Figure 10) |
| `requirements.txt` | Python packages |

## Data

The datasets are not included in this repository. Download them from the
original sources.

**Training and internal test data.** Knee Osteoarthritis Dataset with Severity
(Kaggle): https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity

It contains 8,260 knee radiographs from the Osteoarthritis Initiative with a
predefined split. Keep the folders as they are:

```
data/
  train/0 ... train/4      5,778 images
  val/0   ... val/4          826 images
  test/0  ... test/4       1,656 images
```

The folder name is the KL grade. The first seven digits of each file name are
the patient identifier (for example `9001695L.png`). The split is at patient
level: 2,889 / 413 / 828 patients with no patient shared between subsets. The
last cell of `02_analysis.ipynb` checks this.

**External data.** Digital Knee X-ray Images (Mendeley Data):
https://doi.org/10.17632/t9ndx37v5h.1

Keep the two expert folders as they are:

```
external_data/
  MedicalExpert-I/0Normal ... 4Severe
  MedicalExpert-II/0Normal ... 4Severe
```

## Setup

Python 3.11 was used. Training was run on Google Colab with an NVIDIA T4 GPU
and PyTorch 2.11. The analysis notebooks run on CPU.

```
pip install -r requirements.txt
```

## How to run

### 1. Train the models

Open `01_train.ipynb`. In the first cell set

```python
MODEL    = 'baseline'      # vanilla | baseline | ordinal | attention | full
BACKBONE = 'resnet18'      # resnet18 | densenet121
SEED     = 0               # 0 | 1 | 2
```

and the path to the data. Run all cells. One run takes about 30 minutes on a
T4 GPU. If the session stops, running the notebook again resumes from the last
finished epoch.

| `MODEL` | Name in the paper |
|---|---|
| `vanilla` | Softmax CNN |
| `baseline` | ProtoNet |
| `ordinal` | ProtoNet + Ordinal |
| `attention` | ProtoNet + CBAM |
| `full` | ProtoNet + Ordinal + CBAM |

Each run writes a folder `runs/<model>_<backbone>_seed<seed>/` containing

| File | Content |
|---|---|
| `best_model.pth` | Weights of the checkpoint with the best validation macro-F1 |
| `config.json` | Settings of the run |
| `history.json` | Loss and validation metrics per epoch |
| `results.json` | Test metrics |
| `outputs.npz` | Embeddings, labels, logits, probabilities and predictions for the validation and test sets, training embeddings, and the class prototypes |

The paper uses all 30 combinations (5 models x 2 backbones x 3 seeds).

### 2. Internal analysis

Open `02_analysis.ipynb`, set `ROOT` in the first cell to the folder that
contains `runs/`, and run all cells. The first cell stops with a message if
any of the 30 runs is missing. Tables are written to `tables/` and figures to
`figures/`.

### 3. External validation

Open `03_external_validation.ipynb` and set `EXT_ROOT` (external dataset),
`REVISION` (folder that contains `runs/`) and `KAGGLE_TEST` (the `test` folder
of the Kaggle data). Run all cells. Results are written to `external/`.

The notebook removes exact duplicate images, images with conflicting labels,
images showing both knees, and near-duplicate images before evaluation. The
final evaluation set has 1,292 images.

## Method in brief

- **Backbones:** ResNet-18 and DenseNet-121, pretrained on ImageNet.
- **Prototypical networks:** episodic training, 5-way, 5 support and 5 query
  images per grade, 115 episodes per epoch.
- **Softmax baseline:** same backbone with a linear layer, batch size 32.
- **Training:** 30 epochs, Adam (learning rate 1e-4), ReduceLROnPlateau
  (factor 0.5, patience 5), no early stopping.
- **Evaluation:** class prototypes are the mean embeddings of the training
  images of each grade. Validation and test labels are never used to build
  prototypes. The checkpoint is selected by validation macro-F1.
- **Statistics:** mean and standard deviation over three seeds, paired
  bootstrap with 2,000 resamples, Holm correction, exact McNemar test.

## Where each table and figure comes from

| Paper | File | Notebook |
|---|---|---|
| Table 1 | `tables/table_dataset_distribution.csv` | 02 |
| Tables 3, 4 | `tables/table_main_results.csv` | 02 |
| Tables 5, 8 | `tables/statistical_comparisons.csv` | 02 |
| Table 6 | `tables/per_grade_recall.csv` | 02 |
| Table 7 | `tables/table_auc.csv` | 02 |
| Table 9 | `tables/logit_adjustment_all_runs.csv` | 02 |
| Table 10 | `external/table_external_results.csv` | 03 |
| Table 11 | `external/external_adaptation_linear_probe.csv` | 03 |
| Figure 5 | `figures/fig_per_grade_f1` | 02 |
| Figure 6 | `figures/fig_confusion_softmax_vs_protonet` | 02 |
| Figure 7 | `figures/fig_kshot` | 02 |
| Figure 8 | `figures/fig_training_curves` | 02 |
| Figure 9 | `figures/fig_tsne` | 02 |
| Figure 10 | `figures/fig_external_kshot` | 03 |

Other files in `tables/` and `external/` hold per-run values and the numbers
quoted in the text (ordinal loss activity, prototype ordering, K-shot results,
bootstrap comparisons).

## Trained models

The 30 trained models are not stored in this repository because of their size.
They are available from the authors on request.

## Citation

If you use this code, please cite the paper. The full reference will be added
after publication.

## Contact

Md. Nyem Hasan Bhuiyan 
