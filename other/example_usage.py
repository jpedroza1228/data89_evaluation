"""
Example usage of the Data89 Evaluation Pipeline

This script demonstrates various ways to use the evaluation pipeline.
"""

from evaluation_pipeline import EvaluationPipeline, ModelRunner, StudentMasteryAnalyzer
import numpy as np

# Example 1: Run with mock data (for testing)
def example_mock_data():
    """Run pipeline with automatically generated mock data."""
    print("Example 1: Running with mock data")
    print("-" * 60)
    
    pipeline = EvaluationPipeline()
    pipeline.run_all_models(use_mock=True)


# Example 2: Run with Qualtrics API
def example_qualtrics_api():
    """Run pipeline with data from Qualtrics API."""
    print("\nExample 2: Running with Qualtrics API")
    print("-" * 60)
    
    # Option A: Use environment variables (recommended)
    pipeline = EvaluationPipeline()
    
    # Option B: Pass credentials directly
    # pipeline = EvaluationPipeline(
    #     api_token="your_token",
    #     datacenter="your_datacenter",
    #     survey_id="your_survey_id"
    # )
    
    pipeline.run_all_models(use_mock=False)


# Example 3: Run with custom data
def example_custom_data():
    """Run pipeline with custom response data."""
    print("\nExample 3: Running with custom data")
    print("-" * 60)
    
    # Prepare your own data
    n_students = 50
    n_items = 15
    n_attributes = 3
    
    # Create response matrix (students × items)
    # 1 = correct, 0 = incorrect
    response_matrix = np.random.binomial(1, 0.65, size=(n_students, n_items))
    
    # Create Q-matrix (items × attributes)
    # Indicates which attributes each item measures
    q_matrix = np.array([
        [1, 0, 0],  # Item 1 measures attribute 1
        [1, 0, 0],
        [1, 1, 0],  # Item 3 measures attributes 1 and 2
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [1, 1, 1],
    ])
    
    # Student IDs
    student_ids = [f"STU{i:04d}" for i in range(1, n_students + 1)]
    
    custom_data = {
        'N': n_students,
        'J': n_items,
        'K': n_attributes,
        'Y': response_matrix,
        'Q': q_matrix,
        'max_pattern': 2 ** n_attributes,  # 2^K possible attribute patterns
        'student_ids': student_ids
    }
    
    pipeline = EvaluationPipeline()
    pipeline.run_all_models(data=custom_data)


# Example 4: Run individual models
def example_individual_models():
    """Run specific models individually."""
    print("\nExample 4: Running individual models")
    print("-" * 60)
    
    # Prepare data
    n_students = 50
    n_items = 15
    response_matrix = np.random.binomial(1, 0.6, size=(n_students, n_items))
    student_ids = [f"STU{i:04d}" for i in range(1, n_students + 1)]
    
    # Initialize components
    model_runner = ModelRunner()
    mastery_analyzer = StudentMasteryAnalyzer(mastery_threshold=0.65)
    
    # Run a specific 2PL model
    print("\nRunning 2PL Model 1...")
    data_2pl = {
        'N': n_students,
        'J': n_items,
        'Y': response_matrix
    }
    
    fit = model_runner.run_twopl_model(1, data_2pl)
    
    if fit is not None:
        # Analyze results
        results = mastery_analyzer.analyze_twopl_mastery(fit, student_ids)
        
        # Save flagged students
        mastery_analyzer.save_flagged_students(results, "twopl_model_1_custom")
        
        # Print summary
        print(f"\nResults Summary:")
        print(f"Total students: {len(results)}")
        print(f"Flagged students: {results['flagged'].sum()}")
        print(f"Average ability: {results['ability_estimate'].mean():.3f}")
        print(f"\nTop 5 students by ability:")
        print(results.nlargest(5, 'ability_estimate')[['student_id', 'ability_estimate', 'mastery_probability']])
        print(f"\nBottom 5 students by ability:")
        print(results.nsmallest(5, 'ability_estimate')[['student_id', 'ability_estimate', 'mastery_probability']])


# Example 5: Custom mastery threshold
def example_custom_threshold():
    """Use a custom mastery threshold."""
    print("\nExample 5: Custom mastery threshold")
    print("-" * 60)
    
    # Create analyzer with 60% threshold instead of default 70%
    mastery_analyzer = StudentMasteryAnalyzer(mastery_threshold=0.6)
    
    # You can also modify it in the pipeline
    pipeline = EvaluationPipeline()
    pipeline.mastery_analyzer = mastery_analyzer
    
    # Run with lower threshold
    pipeline.run_all_models(use_mock=True)


# Example 6: Process specific student subgroups
def example_subgroup_analysis():
    """Analyze specific student subgroups."""
    print("\nExample 6: Subgroup analysis")
    print("-" * 60)
    
    # Simulate data for two groups
    n_group1 = 30  # High performers
    n_group2 = 20  # Lower performers
    n_items = 15
    
    # Different performance levels
    responses_group1 = np.random.binomial(1, 0.8, size=(n_group1, n_items))
    responses_group2 = np.random.binomial(1, 0.5, size=(n_group2, n_items))
    
    # Combine
    all_responses = np.vstack([responses_group1, responses_group2])
    student_ids = [f"HIGH_{i:03d}" for i in range(n_group1)] + \
                  [f"LOW_{i:03d}" for i in range(n_group2)]
    
    # Run analysis
    model_runner = ModelRunner()
    mastery_analyzer = StudentMasteryAnalyzer()
    
    data = {
        'N': n_group1 + n_group2,
        'J': n_items,
        'Y': all_responses
    }
    
    fit = model_runner.run_twopl_model(1, data)
    
    if fit is not None:
        results = mastery_analyzer.analyze_twopl_mastery(fit, student_ids)
        
        # Compare groups
        results['group'] = results['student_id'].str.split('_').str[0]
        
        print("\nGroup comparison:")
        print(results.groupby('group')[['ability_estimate', 'mastery_probability', 'flagged']].agg({
            'ability_estimate': ['mean', 'std'],
            'mastery_probability': ['mean', 'std'],
            'flagged': 'sum'
        }))


if __name__ == "__main__":
    # Run different examples
    # Uncomment the example you want to run
    
    # Basic examples
    example_mock_data()
    # example_qualtrics_api()  # Requires API credentials
    # example_custom_data()
    
    # Advanced examples
    # example_individual_models()
    # example_custom_threshold()
    # example_subgroup_analysis()
