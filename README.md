# data89_evaluation

This repo houses the pipeline to analyze weekly assessment data along with the bayesian diagnostic models to assess student mastery after each course.

## Overview

The Data89 Evaluation system provides comprehensive Bayesian diagnostic modeling for educational assessment data. It includes:

- **7 GDINA Models**: General Diagnostic Classification Models for attribute mastery assessment
- **7 2PL IRT Models**: Two-Parameter Logistic Item Response Theory models for ability estimation
- **Qualtrics Integration**: Automated data retrieval from Qualtrics surveys
- **HTML Reporting**: Comprehensive diagnostic reports with model findings
- **Student Flagging**: Automated identification of students requiring intervention

## Project Structure

```
data89_evaluation/
├── stan_models/              # Stan model files
│   ├── gdina_model_1.stan   # Basic GDINA model
│   ├── gdina_model_2.stan   # GDINA with hierarchical priors
│   ├── gdina_model_3.stan   # GDINA with covariates
│   ├── gdina_model_4.stan   # GDINA with item difficulty
│   ├── gdina_model_5.stan   # GDINA with slipping/guessing
│   ├── gdina_model_6.stan   # GDINA with temporal effects
│   ├── gdina_model_7.stan   # GDINA with random effects
│   ├── twopl_model_1.stan   # Basic 2PL IRT model
│   ├── twopl_model_2.stan   # 2PL with hierarchical priors
│   ├── twopl_model_3.stan   # 2PL with student covariates
│   ├── twopl_model_4.stan   # 2PL with item groups
│   ├── twopl_model_5.stan   # 2PL with temporal effects
│   ├── twopl_model_6.stan   # 2PL multidimensional
│   └── twopl_model_7.stan   # 2PL with testlet effects
├── student_data/             # Output CSV files of flagged students
├── evaluation_pipeline.py    # Main pipeline script
└── requirements.txt          # Python dependencies
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install CmdStan (required for Stan models):
```bash
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

## Configuration

### Qualtrics API Setup

Set environment variables for Qualtrics API access:

```bash
export QUALTRICS_API_TOKEN="your_api_token"
export QUALTRICS_DATACENTER="your_datacenter_id"
export QUALTRICS_SURVEY_ID="your_survey_id"
```

Or create a `.env` file:
```
QUALTRICS_API_TOKEN=your_api_token
QUALTRICS_DATACENTER=your_datacenter_id
QUALTRICS_SURVEY_ID=your_survey_id
```

## Usage

### Running the Pipeline

```bash
python evaluation_pipeline.py
```

This will:
1. Connect to Qualtrics API and retrieve survey data (or use mock data if not configured)
2. Run all 7 GDINA models
3. Run all 7 2PL IRT models
4. Generate CSV files of flagged students in `student_data/`
5. Create an HTML diagnostic report: `model_diagnostics_report.html`

### Custom Usage

```python
from evaluation_pipeline import EvaluationPipeline

# Initialize pipeline
pipeline = EvaluationPipeline(
    api_token="your_token",
    datacenter="your_datacenter",
    survey_id="your_survey_id"
)

# Run with real data
pipeline.run_all_models(use_mock=False)

# Or provide custom data
custom_data = {
    'N': 100,  # number of students
    'J': 20,   # number of items
    'K': 3,    # number of attributes
    'Y': response_matrix,  # N x J binary matrix
    'Q': q_matrix,         # J x K binary Q-matrix
    'max_pattern': 8,      # 2^K
    'student_ids': student_list
}
pipeline.run_all_models(data=custom_data)
```

## Models

### GDINA Models

1. **Model 1**: Basic GDINA model with standard priors
2. **Model 2**: Hierarchical structure for attribute patterns
3. **Model 3**: Incorporates student-level covariates
4. **Model 4**: Includes item difficulty parameters
5. **Model 5**: Explicitly models slipping and guessing
6. **Model 6**: Accounts for temporal/learning effects
7. **Model 7**: Includes random effects for items

### 2PL IRT Models

1. **Model 1**: Basic 2PL with discrimination and difficulty
2. **Model 2**: Hierarchical priors for item parameters
3. **Model 3**: Student-level covariates affecting ability
4. **Model 4**: Item grouping by content area
5. **Model 5**: Longitudinal/growth modeling
6. **Model 6**: Multidimensional abilities
7. **Model 7**: Testlet effects for local item dependence

## Output

### Student Data CSV Files

Flagged students are saved to `student_data/` with filenames like:
- `gdina_model_1_flagged_students.csv`
- `twopl_model_1_flagged_students.csv`

Each file contains:
- `student_id`: Student identifier
- `mastery_probability`: Estimated probability of mastery
- `mastery_status`: Boolean mastery classification
- `flagged`: Whether student is flagged for intervention
- Additional model-specific metrics

### HTML Diagnostic Report

The report (`model_diagnostics_report.html`) includes:
- Model convergence diagnostics
- Parameter summaries
- Number and percentage of flagged students
- Mastery probability distributions
- Model-specific findings

## Mastery Threshold

Students are flagged when their estimated mastery probability falls below **0.7** (70%). This threshold can be adjusted:

```python
from evaluation_pipeline import StudentMasteryAnalyzer

analyzer = StudentMasteryAnalyzer(mastery_threshold=0.6)  # 60% threshold
```

## License

See LICENSE file for details.
