# PIGuard Extension

Computer Security team project: reproducing and extending
[PIGuard (ACL 2025)](https://aclanthology.org/2025.acl-long.1468/).

## Team

- 이석태 (Team Captain): project management, report
- 권수현 (Developer): reproduction, Korean test set evaluation, code package
- 이승희 (Presenter): baseline comparison, demo, slides

## Structure

| Path | Contents |
| --- | --- |
| `PIGuard/` | Original code ([leolee99/PIGuard](https://github.com/leolee99/PIGuard), MIT License, commit `1b5751e`) |
| `extension/korean_testset/` | Korean over-defense test set |
| `extension/baselines/` | Baseline comparison |
| `results/` | Result numbers and figures |
| `demo/` | Demo |

Keep `PIGuard/` as the unmodified original where possible, and put our changes in the other folders.

## Do not commit

- PINT dataset: it is not public
- Model checkpoints: use the [Hugging Face model](https://huggingface.co/leolee99/PIGuard)

## Install and run

### Installation

```bash
git clone https://github.com/seoktae-lee/piguard-extension.git
cd piguard-extension
pip install -r PIGuard/requirements.txt
```

### Model

- Hugging Face: leolee99/PIGuard
- Evaluation environment: Google Colab
- Original source code is preserved in PIGuard/.

### Evaluation Results

The evaluation results are available in the results/ directory.

```bash
python -m json.tool results/Korean_Final_Results.json
```

The Korean benign evaluation datasets are available in
extension/korean_testset/.

Raw attack samples are excluded from the public repository.
The released result files provide aggregate evaluation scores.



## Korean Extension Evaluation

### Experiment Environment

- Google Colab
- Model: leolee99/PIGuard
- Language: Korean
- Total evaluation samples: 390
- Benign samples: 270
- Injection samples: 120
- Model fine-tuning: Not performed

### Original Reproduction Results

| Dataset | Accuracy |
| --- | ---: |
| WildGuard | 76.11% |
| BIPIA Text | 38.67% |
| BIPIA Code | 98.00% |
| BIPIA Overall | 68.33% |
| NotInject Overall | 88.50% |
| Macro Average (without PINT) | 77.65% |

### Korean Evaluation Results

| Metric | Result |
| --- | ---: |
| Benign Classification Accuracy | 100.00% |
| Attack Detection Rate | 33.33% |
| Overall Accuracy | 79.49% |
| Precision | 100.00% |
| Recall | 33.33% |
| F1-score | 50.00% |
| Balanced Accuracy | 66.67% |
| False Positive Rate | 0.00% |
| False Negative Rate | 66.67% |

### Confusion Matrix

| Actual / Predicted | Benign | Injection |
| --- | ---: | ---: |
| Benign | 270 | 0 |
| Injection | 80 | 40 |

### Limitations

- The Korean dataset consists of synthetic samples.
- Human review of dataset labels is still required.
- The original model was evaluated without fine-tuning.
- PINT was excluded from the reproduction experiment.
- Original and Korean accuracy scores are not directly comparable.
- Raw attack samples are not included in the public repository.
- Full attack evaluation cannot be reproduced using only the released files.

### Result Files

- results/Original_Reproduction_Results.json
- results/Korean_Benign_Results.json
- results/Korean_Attack_Results.json
- results/Korean_Final_Results.json
