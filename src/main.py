from pathlib import Path
from data_processing import (
    load_diseases_data, 
    analyze_diseases, 
    select_disease, 
    create_disease_network
)
from visualization import (
    plot_disease_distribution, 
    plot_disease_hierarchy,  # Changed from plot_score_distribution
    visualize_disease_network,
    plot_training_history,
    plot_prediction_analysis
)
from model_builder import DiseasePredictor
import numpy as np
import torch
from sklearn.metrics import classification_report

def main():
    # Set up paths
    base_dir = Path(__file__).parent.parent
    diseases_dir = base_dir / 'diseases'
    models_dir = base_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Load and analyze data
    print("Loading diseases data...")
    df = load_diseases_data(diseases_dir)
    
    print("\nAnalyzing diseases...")
    disease_stats, disease_counts = analyze_diseases(df)
    
    # Print basic statistics
    print("\nDisease Statistics:")
    for key, value in disease_stats.items():
        print(f"{key}: {value}")
    
    # Select a disease
    print("\nSelecting target disease...")
    selected_disease = select_disease(disease_counts)
    print(f"Selected disease: {selected_disease['disease_id']}")  # Changed from 'diseaseId'
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_disease_distribution(disease_counts)
    plot_disease_hierarchy(df)  # Changed from plot_score_distribution
    
    # Initialize prediction model
    print("\nPreparing model...")
    predictor = DiseasePredictor(input_size=7)  # 7 features from prepare_features
    
    # Prepare features and labels
    print("Preparing features...")
    features = predictor.prepare_features(df)
    # For demonstration, we'll create synthetic labels (0 or 1)
    # In real application, these would come from experimental data
    labels = np.random.binomial(1, 0.3, size=len(features))
    
    # Prepare data loaders
    train_loader, val_loader = predictor.prepare_data(
        features.values, 
        labels, 
        batch_size=32
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train model
    history = predictor.train_model(
        train_loader, 
        val_loader,
        epochs=50,
        lr=0.001,
        device=device
    )
    
    # Save the trained model
    model_path = models_dir / 'disease_predictor.pt'
    predictor.save_model(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Plot training history
    plot_training_history(history)
    
    # Get predictions for analysis
    predictor.eval()
    with torch.no_grad():
        all_features = torch.FloatTensor(
            predictor.scaler.transform(features.values)
        ).to(device)
        predictions = predictor(all_features).cpu().numpy()
    
    # Plot prediction analysis
    y_pred = (predictions > 0.5).astype(int)
    plot_prediction_analysis(labels, y_pred, predictions)
    
    # Print model performance metrics
    print("\nModel Performance Metrics:")
    report = classification_report(labels, y_pred)
    print(report)
    
    # Create and visualize network
    print("\nCreating disease network...")
    G = create_disease_network(df, selected_disease['disease_id'])  # Changed from 'diseaseId'
    visualize_disease_network(G)
    
    print("\nAnalysis complete. Check the output files for visualizations.")

if __name__ == "__main__":
    main()
