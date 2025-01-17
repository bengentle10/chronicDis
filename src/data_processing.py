import pandas as pd
import networkx as nx
from pathlib import Path

def load_diseases_data(diseases_dir):
    """Load diseases data from all Parquet files in the directory."""
    diseases_path = Path(diseases_dir)
    
    try:
        df = pd.read_parquet(diseases_path, engine='pyarrow')
        print("\nDataset Info:")
        print(df.info())
        print("\nColumns available:")
        print(df.columns.tolist())
        return df
    except Exception as e:
        raise ImportError(f"Failed to read Parquet files. Error: {str(e)}") from e

def analyze_diseases(df):
    """Analyze diseases distribution and characteristics."""
    disease_stats = {
        'total_diseases': len(df),
        'diseases_with_description': df['description'].notna().sum(),
        'diseases_with_synonyms': df['synonyms'].notna().sum(),
        'diseases_by_ontology': df['ontology'].apply(lambda x: x.get('isTherapeuticArea', False)).value_counts().to_dict()
    }
    
    # Analyze disease hierarchy
    disease_counts = pd.DataFrame({
        'disease_id': df['id'],
        'name': df['name'],
        'child_count': df['children'].str.len(),
        'parent_count': df['parents'].str.len(),
        'is_leaf': df['ontology'].apply(lambda x: x.get('leaf', False))
    })
    
    return disease_stats, disease_counts

def select_disease(disease_counts, min_children=5):
    """Select a disease based on its position in the hierarchy."""
    # Select diseases that have enough children (subdiseases)
    filtered_diseases = disease_counts[
        (disease_counts['child_count'] >= min_children) &
        (~disease_counts['is_leaf'])  # Not a leaf node
    ]
    
    # Sort by number of children to find major disease categories
    sorted_diseases = filtered_diseases.sort_values(
        by=['child_count', 'parent_count'],
        ascending=[False, True]
    )
    
    return sorted_diseases.iloc[0]

def create_disease_network(df, selected_disease_id):
    """Create a network showing disease hierarchy."""
    G = nx.DiGraph()
    
    # Find the selected disease and its direct children
    disease_row = df[df['id'] == selected_disease_id].iloc[0]
    
    # Add the main disease node
    G.add_node(selected_disease_id, 
               name=disease_row['name'],
               type='main')
    
    # Add child diseases
    for child_id in disease_row['children']:
        child_data = df[df['id'] == child_id]
        if not child_data.empty:
            G.add_node(child_id,
                      name=child_data.iloc[0]['name'],
                      type='child')
            G.add_edge(selected_disease_id, child_id)
    
    return G
