# Human Assessment Module
This repository provides the code to support the creation of personalised machine learning models to quantify the cognitive performance and using psycophysiological data. It is organised into four main components:

- **Framework for Experimental Protocol and Data Collection**
- **Personalised Quantification of Cognitive Performance**


## 🧪 Framework for Experimental Protocol and Data Collection
The folder `reaction_time_protocol` contains the MATLAB experimental framework used to gather physiological and behavioral data, following the experimental design described in [Rodrigues et al., 2018](https://www.mdpi.com/1660-4601/15/6/1080).

### Protocol Overview
The experimental session includes the following tests:
- `Baseline`: Participants sit for 10 minutes to establish resting physiological signals.
- `2-Choice Reaction Time Task (CRTT)`: A selective-attention task, where participants identified either the large, global letter or the small, local letters of a hierarchically organized visual object. Their response time and correct/incorrect/missed answers were recorded.
- `Trier Social Stress Test (TSST)`: A validated acute psychosocial stress paradigm involving public speaking and mental arithmetic tasks in front of an evaluative panel.

Order of Tasks:  
`Baseline → CRTT1 → TSST → CRTT2`

Along the protocol, various **psychological self-report scales** were used and their results are saved in the created raw dataset.

### Outputs
The experimental framework collects the following data per participant:
- **ECG:** Raw electrocardiogram (ECG) signal sampled during all protocol phases.
- **VAS:** Self-reported stress level using the Visual Analogue Scale.
- **STAI_6items:** Score from the 6-item short form of the State-Trait Anxiety Inventory, assessing current anxiety levels.
- **Right_answers:** The correct/expected responses for the CRTT task.
- **Answers:** Participant responses.
- **Answer_reaction_time:** Reaction time (in seconds) for each stimulus in the CRTT task.
- **Answer_timing:** Timestamps indicating when each visual stimulus was presented (used to align with physiological signals).

### Dependencies
*[Psychtoolbox](http://psychtoolbox.org/) must be installed and tested before running the visual tasks.*

## 🧠 Personalised Quantification of Cognitive Performance

### Input
Each personalised model expects a JSON file as input containing rows of extracted physiological features. The expected structure is as follows:

**File Type**: `.json`

**Keys**:

### Protocol Variables
| Key Name             | Description |
|-------------------------|-------------|
| **Test_phase**          | The phase of the experiment (e.g., baseline, CRTT1, TSST, CRTT2). |
| **ECG**                 | Raw ECG signal, 1 lead & Fs = 500 HZ. |
| **VAS**                 | Visual Analogue Scale rating. |
| **STAI_6items**         | State-Trait Anxiety Inventory (6-item version) score. |
| **Accuracy**            | Percentage of correct responses in a cognitive task. |
| **Reaction_time**       | Response time in seconds. |
| **RT_std**              | Standard deviation of reaction times. |
| **Cognitive_performance** | Composite score of cognitive task performance. |

<sub>Cognitive Performance = Accuracy / (Average Reaction Time × Reaction Time Std Dev)</sub>


### ECG Waveform Features
| Column Name             | Description |
|-------------------------|-------------|
| **p_wave_duration**     | Duration of the P wave (ms). |
| **pr_interval**         | Time from the start of the P wave to the start of the QRS complex (ms). |
| **pr_segment**          | Time between the end of the P wave and the start of the QRS complex (ms). |
| **qrs_duration**        | Duration of the QRS complex (ms). |
| **qt_interval**         | Time between the start of the Q wave and the end of the T wave (ms). |
| **st_segment**          | Segment between the QRS complex and the T wave (ms). |
| **st_interval**         | Time between the J point and the end of the T wave (ms). |
| **t_wave_duration**     | Duration of the T wave (ms). |
| **tp_segment**          | Time from the end of the T wave to the start of the next P wave (ms). |
| **rr_interval**         | Time between two consecutive R-peaks (ms). |

### Heart Rate Variability (HRV) Metrics
| Column Name             | Description |
|-------------------------|-------------|
| **mean_nni**           | Mean of normal-to-normal (NN) intervals (ms). |
| **sdnn**               | Standard deviation of NN intervals (ms). |
| **sdsd**               | Standard deviation of successive differences between NN intervals (ms). |
| **nni_50**             | Number of pairs of successive NN intervals differing by more than 50 ms. |
| **pnni_50**            | Percentage of NN50 count divided by the total number of NN intervals. |
| **nni_20**             | Number of pairs of successive NN intervals differing by more than 20 ms. |
| **pnni_20**            | Percentage of NN20 count divided by the total number of NN intervals. |
| **rmssd**              | Root mean square of successive differences (ms). |
| **median_nni**         | Median of NN intervals (ms). |
| **range_nni**          | Range of NN intervals (max-min) (ms). |
| **cvsd**               | Coefficient of variation of successive differences. |
| **cvnni**              | Coefficient of variation of NN intervals. |

### Heart Rate Metrics
| Column Name             | Description |
|-------------------------|-------------|
| **mean_hr**            | Mean heart rate (beats per minute). |
| **max_hr**             | Maximum heart rate recorded. |
| **min_hr**             | Minimum heart rate recorded. |
| **std_hr**             | Standard deviation of heart rate. |

### Frequency-Domain HRV Metrics
| Column Name             | Description |
|-------------------------|-------------|
| **lf**                 | Low-frequency power (ms²). |
| **hf**                 | High-frequency power (ms²). |
| **lf_hf_ratio**        | Ratio of LF to HF power. |
| **lfnu**               | Low-frequency power in normalized units. |
| **hfnu**               | High-frequency power in normalized units. |
| **total_power**        | Total spectral power of HRV (ms²). |
| **vlf**                | Very low-frequency power (ms²). |
  
# Running the Code
To execute the algorithm, ensure you are in the root directory of the repository. The script should be run using the following command:

```bash
python -m code.algorithms.cognition.model_personalization
```

Upon execution, you will be prompted to select a file containing the dataset. 

⚠ Attention: The input dataset must follow the required structure, though no example is currently provided, as the data used for testing is private.

### Output Details:
- Running the command will generate a model based on the provided dataset.
- Users must enter a unique code to the model when prompted.
- The model will be saved as a `.pkl`file in the `personalised_models` folder within the working directory.  - The - The file will follow this naming convention: `personalised_models/model_<user_code>.pkl`

