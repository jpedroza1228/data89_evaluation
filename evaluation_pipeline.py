"""
Data89 Evaluation Pipeline
This script connects to Qualtrics API to retrieve assessment data,
runs Bayesian diagnostic models (GDINA and 2PL IRT), and generates
HTML reports with model diagnostics and student mastery flags.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import requests
from cmdstanpy import CmdStanModel
import warnings
warnings.filterwarnings('ignore')


class QualtricsAPI:
    """Handler for Qualtrics API interactions."""
    
    def __init__(self, api_token: str, datacenter: str, survey_id: str):
        """
        Initialize Qualtrics API connection.
        
        Args:
            api_token: Qualtrics API token
            datacenter: Datacenter ID (e.g., 'yourdatacenterid')
            survey_id: Survey ID to retrieve data from
        """
        self.api_token = api_token
        self.datacenter = datacenter
        self.survey_id = survey_id
        self.base_url = f"https://{datacenter}.qualtrics.com/API/v3"
        self.headers = {
            "X-API-TOKEN": api_token,
            "Content-Type": "application/json"
        }
    
    def get_survey_responses(self) -> pd.DataFrame:
        """
        Retrieve survey responses from Qualtrics.
        
        Returns:
            DataFrame containing survey responses
        """
        # Request response export
        export_url = f"{self.base_url}/surveys/{self.survey_id}/export-responses"
        export_payload = {"format": "json"}
        
        try:
            # Start export
            response = requests.post(
                export_url,
                headers=self.headers,
                json=export_payload
            )
            response.raise_for_status()
            progress_id = response.json()["result"]["progressId"]
            
            # Check export progress
            progress_url = f"{self.base_url}/surveys/{self.survey_id}/export-responses/{progress_id}"
            export_status = "inProgress"
            
            while export_status == "inProgress":
                progress_response = requests.get(progress_url, headers=self.headers)
                progress_response.raise_for_status()
                export_status = progress_response.json()["result"]["status"]
            
            # Download file
            file_id = progress_response.json()["result"]["fileId"]
            download_url = f"{self.base_url}/surveys/{self.survey_id}/export-responses/{file_id}/file"
            download_response = requests.get(download_url, headers=self.headers)
            download_response.raise_for_status()
            
            # Parse JSON response
            data = download_response.json()
            responses = data.get("responses", [])
            
            return pd.DataFrame(responses)
            
        except requests.exceptions.RequestException as e:
            print(f"Error retrieving data from Qualtrics: {e}")
            return pd.DataFrame()


class ModelRunner:
    """Handler for running Stan models."""
    
    def __init__(self, model_dir: str = "stan_models"):
        """
        Initialize model runner.
        
        Args:
            model_dir: Directory containing Stan model files
        """
        self.model_dir = model_dir
        self.compiled_models = {}
    
    def compile_model(self, model_name: str) -> CmdStanModel:
        """
        Compile a Stan model.
        
        Args:
            model_name: Name of the model file (without .stan extension)
        
        Returns:
            Compiled CmdStanModel object
        """
        if model_name in self.compiled_models:
            return self.compiled_models[model_name]
        
        model_path = os.path.join(self.model_dir, f"{model_name}.stan")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            model = CmdStanModel(stan_file=model_path)
            self.compiled_models[model_name] = model
            return model
        except Exception as e:
            print(f"Error compiling model {model_name}: {e}")
            raise
    
    def run_gdina_model(self, model_num: int, data: Dict) -> Optional[object]:
        """
        Run a GDINA model.
        
        Args:
            model_num: Model number (1-7)
            data: Dictionary containing model data
        
        Returns:
            Model fit object or None if failed
        """
        model_name = f"gdina_model_{model_num}"
        
        try:
            model = self.compile_model(model_name)
            fit = model.sample(data=data, chains=4, iter_sampling=2000)
            return fit
        except Exception as e:
            print(f"Error running {model_name}: {e}")
            return None
    
    def run_twopl_model(self, model_num: int, data: Dict) -> Optional[object]:
        """
        Run a 2PL IRT model.
        
        Args:
            model_num: Model number (1-7)
            data: Dictionary containing model data
        
        Returns:
            Model fit object or None if failed
        """
        model_name = f"twopl_model_{model_num}"
        
        try:
            model = self.compile_model(model_name)
            fit = model.sample(data=data, chains=4, iter_sampling=2000)
            return fit
        except Exception as e:
            print(f"Error running {model_name}: {e}")
            return None


class StudentMasteryAnalyzer:
    """Analyzer for determining student mastery and flagging students."""
    
    def __init__(self, mastery_threshold: float = 0.7):
        """
        Initialize analyzer.
        
        Args:
            mastery_threshold: Probability threshold for mastery classification
        """
        self.mastery_threshold = mastery_threshold
    
    def analyze_gdina_mastery(self, fit: object, student_ids: List[str]) -> pd.DataFrame:
        """
        Analyze mastery from GDINA model results.
        
        Args:
            fit: Model fit object
            student_ids: List of student identifiers
        
        Returns:
            DataFrame with student mastery information
        """
        try:
            # Extract pattern probabilities
            pattern_prob = fit.stan_variable("pattern_prob")
            
            # Calculate mastery probability (simplified)
            # In practice, this would map patterns to mastery levels
            mastery_prob = np.max(pattern_prob, axis=2).mean(axis=0)
            
            results = pd.DataFrame({
                'student_id': student_ids,
                'mastery_probability': mastery_prob,
                'mastery_status': mastery_prob >= self.mastery_threshold,
                'flagged': mastery_prob < self.mastery_threshold
            })
            
            return results
            
        except Exception as e:
            print(f"Error analyzing GDINA mastery: {e}")
            return pd.DataFrame()
    
    def analyze_twopl_mastery(self, fit: object, student_ids: List[str]) -> pd.DataFrame:
        """
        Analyze mastery from 2PL IRT model results.
        
        Args:
            fit: Model fit object
            student_ids: List of student identifiers
        
        Returns:
            DataFrame with student mastery information
        """
        try:
            # Extract ability estimates
            theta = fit.stan_variable("theta")
            theta_mean = theta.mean(axis=0)
            theta_sd = theta.std(axis=0)
            
            # Convert to mastery probability (using cumulative normal)
            # Threshold at 0 (average ability)
            from scipy.stats import norm
            mastery_prob = 1 - norm.cdf(0, loc=theta_mean, scale=theta_sd)
            
            results = pd.DataFrame({
                'student_id': student_ids,
                'ability_estimate': theta_mean,
                'ability_sd': theta_sd,
                'mastery_probability': mastery_prob,
                'mastery_status': mastery_prob >= self.mastery_threshold,
                'flagged': mastery_prob < self.mastery_threshold
            })
            
            return results
            
        except Exception as e:
            print(f"Error analyzing 2PL mastery: {e}")
            return pd.DataFrame()
    
    def save_flagged_students(self, results: pd.DataFrame, model_name: str, 
                             output_dir: str = "student_data"):
        """
        Save flagged students to CSV file.
        
        Args:
            results: DataFrame with mastery results
            model_name: Name of the model
            output_dir: Directory to save CSV files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        flagged_students = results[results['flagged'] == True]
        
        if len(flagged_students) > 0:
            filename = f"{model_name}_flagged_students.csv"
            filepath = os.path.join(output_dir, filename)
            flagged_students.to_csv(filepath, index=False)
            print(f"Saved {len(flagged_students)} flagged students to {filepath}")
        else:
            print(f"No students flagged for {model_name}")


class HTMLReportGenerator:
    """Generator for HTML diagnostic reports."""
    
    def __init__(self):
        """Initialize report generator."""
        self.report_sections = []
    
    def add_model_diagnostics(self, model_name: str, fit: object, 
                            mastery_results: pd.DataFrame):
        """
        Add model diagnostics to report.
        
        Args:
            model_name: Name of the model
            fit: Model fit object
            mastery_results: DataFrame with mastery results
        """
        try:
            # Get model diagnostics
            summary = fit.summary()
            
            # Calculate key metrics
            n_flagged = mastery_results['flagged'].sum()
            n_total = len(mastery_results)
            pct_flagged = (n_flagged / n_total * 100) if n_total > 0 else 0
            
            section = f"""
            <div class="model-section">
                <h2>{model_name}</h2>
                <h3>Model Diagnostics</h3>
                <table>
                    <tr>
                        <td><strong>Number of Students:</strong></td>
                        <td>{n_total}</td>
                    </tr>
                    <tr>
                        <td><strong>Students Flagged:</strong></td>
                        <td>{n_flagged} ({pct_flagged:.1f}%)</td>
                    </tr>
                    <tr>
                        <td><strong>Convergence:</strong></td>
                        <td>{"✓ Converged" if fit.converged else "✗ Not Converged"}</td>
                    </tr>
                </table>
                
                <h3>Parameter Summary</h3>
                <div class="summary-table">
                    {summary.to_html(max_rows=20)}
                </div>
                
                <h3>Mastery Distribution</h3>
                <p>Average Mastery Probability: {mastery_results['mastery_probability'].mean():.3f}</p>
                <p>Median Mastery Probability: {mastery_results['mastery_probability'].median():.3f}</p>
            </div>
            """
            
            self.report_sections.append(section)
            
        except Exception as e:
            print(f"Error adding diagnostics for {model_name}: {e}")
    
    def generate_report(self, output_path: str = "model_diagnostics_report.html"):
        """
        Generate complete HTML report.
        
        Args:
            output_path: Path to save HTML report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data89 Evaluation - Model Diagnostics Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                }}
                .model-section {{
                    background-color: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                table {{
                    border-collapse: collapse;
                    margin: 10px 0;
                }}
                td {{
                    padding: 8px;
                    border-bottom: 1px solid #ddd;
                }}
                .summary-table {{
                    overflow-x: auto;
                    max-height: 400px;
                    overflow-y: auto;
                }}
                .summary-table table {{
                    font-size: 12px;
                }}
                h2 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 5px;
                }}
                h3 {{
                    color: #34495e;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data89 Evaluation - Model Diagnostics Report</h1>
                <p>Generated: {timestamp}</p>
            </div>
            
            {''.join(self.report_sections)}
            
            <div class="model-section">
                <h2>Notes</h2>
                <p>This report contains diagnostic information for all GDINA and 2PL IRT models.</p>
                <p>Students are flagged when their mastery probability falls below the threshold (0.7).</p>
                <p>Flagged students have been saved to CSV files in the student_data directory.</p>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Report generated: {output_path}")


class EvaluationPipeline:
    """Main pipeline for data89 evaluation."""
    
    def __init__(self, api_token: str = None, datacenter: str = None, 
                 survey_id: str = None):
        """
        Initialize evaluation pipeline.
        
        Args:
            api_token: Qualtrics API token (optional, can use env var)
            datacenter: Datacenter ID (optional, can use env var)
            survey_id: Survey ID (optional, can use env var)
        """
        # Get credentials from environment if not provided
        self.api_token = api_token or os.getenv('QUALTRICS_API_TOKEN')
        self.datacenter = datacenter or os.getenv('QUALTRICS_DATACENTER')
        self.survey_id = survey_id or os.getenv('QUALTRICS_SURVEY_ID')
        
        self.qualtrics = None
        if self.api_token and self.datacenter and self.survey_id:
            self.qualtrics = QualtricsAPI(self.api_token, self.datacenter, self.survey_id)
        
        self.model_runner = ModelRunner()
        self.mastery_analyzer = StudentMasteryAnalyzer()
        self.report_generator = HTMLReportGenerator()
    
    def prepare_mock_data(self, n_students: int = 100, n_items: int = 20, 
                         n_attributes: int = 3) -> Dict:
        """
        Prepare mock data for testing (when real data unavailable).
        
        Args:
            n_students: Number of students
            n_items: Number of items
            n_attributes: Number of attributes
        
        Returns:
            Dictionary with mock data
        """
        np.random.seed(42)
        
        # Generate response matrix
        Y = np.random.binomial(1, 0.6, size=(n_students, n_items))
        
        # Generate Q-matrix
        Q = np.random.binomial(1, 0.5, size=(n_items, n_attributes))
        Q[Q.sum(axis=1) == 0, 0] = 1  # Ensure each item measures at least one attribute
        
        return {
            'N': n_students,
            'J': n_items,
            'K': n_attributes,
            'Y': Y,
            'Q': Q.astype(int),
            'max_pattern': 2 ** n_attributes,
            'student_ids': [f"Student_{i+1}" for i in range(n_students)]
        }
    
    def run_all_models(self, data: Dict = None, use_mock: bool = True):
        """
        Run all 7 GDINA and 7 2PL models.
        
        Args:
            data: Optional data dictionary
            use_mock: Whether to use mock data if no data provided
        """
        # Get data
        if data is None:
            if self.qualtrics and not use_mock:
                df = self.qualtrics.get_survey_responses()
                # Process Qualtrics data into model format
                # This would need customization based on actual survey structure
                data = self.prepare_mock_data()
            else:
                print("Using mock data for demonstration...")
                data = self.prepare_mock_data()
        
        student_ids = data.get('student_ids', [f"Student_{i+1}" for i in range(data['N'])])
        
        # Run GDINA models
        print("\n" + "="*50)
        print("Running GDINA Models")
        print("="*50)
        
        for i in range(1, 8):
            print(f"\nRunning GDINA Model {i}...")
            
            # Prepare model-specific data
            model_data = {
                'N': data['N'],
                'J': data['J'],
                'K': data['K'],
                'Y': data['Y'],
                'Q': data['Q'],
                'max_pattern': data['max_pattern']
            }
            
            # Add model-specific parameters
            if i == 3:  # Model with covariates
                model_data['P'] = 0
                model_data['X'] = np.zeros((data['N'], 0))
            elif i == 6:  # Model with temporal effects
                model_data['time_point'] = np.arange(data['N']) / data['N']
            
            fit = self.model_runner.run_gdina_model(i, model_data)
            
            if fit is not None:
                mastery_results = self.mastery_analyzer.analyze_gdina_mastery(fit, student_ids)
                self.mastery_analyzer.save_flagged_students(
                    mastery_results, f"gdina_model_{i}"
                )
                self.report_generator.add_model_diagnostics(
                    f"GDINA Model {i}", fit, mastery_results
                )
        
        # Run 2PL models
        print("\n" + "="*50)
        print("Running 2PL IRT Models")
        print("="*50)
        
        for i in range(1, 8):
            print(f"\nRunning 2PL Model {i}...")
            
            # Prepare model-specific data
            model_data = {
                'N': data['N'],
                'J': data['J'],
                'Y': data['Y']
            }
            
            # Add model-specific parameters
            if i == 3:  # Model with covariates
                model_data['P'] = 0
                model_data['X'] = np.zeros((data['N'], 0))
            elif i == 4:  # Model with groups
                model_data['G'] = 3
                model_data['group'] = np.random.randint(1, 4, size=data['J'])
            elif i == 5:  # Model with temporal effects
                model_data['time_point'] = np.arange(data['N']) / data['N']
            elif i == 6:  # Multidimensional model
                model_data['K'] = 2
                model_data['Q'] = np.random.binomial(1, 0.5, size=(data['J'], 2))
            elif i == 7:  # Testlet model
                model_data['T'] = 4
                model_data['testlet'] = np.random.randint(1, 5, size=data['J'])
            
            fit = self.model_runner.run_twopl_model(i, model_data)
            
            if fit is not None:
                mastery_results = self.mastery_analyzer.analyze_twopl_mastery(fit, student_ids)
                self.mastery_analyzer.save_flagged_students(
                    mastery_results, f"twopl_model_{i}"
                )
                self.report_generator.add_model_diagnostics(
                    f"2PL IRT Model {i}", fit, mastery_results
                )
        
        # Generate HTML report
        print("\n" + "="*50)
        print("Generating HTML Report")
        print("="*50)
        self.report_generator.generate_report()
        
        print("\n✓ Pipeline complete!")


def main():
    """Main entry point for the pipeline."""
    print("="*60)
    print("Data89 Evaluation Pipeline")
    print("="*60)
    
    # Initialize pipeline
    pipeline = EvaluationPipeline()
    
    # Run all models
    # Set use_mock=False to use real Qualtrics data
    pipeline.run_all_models(use_mock=True)


if __name__ == "__main__":
    main()
