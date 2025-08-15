from analysis.unsupervised import cluster_users
from utils.export import dataframe_to_json
import json

if __name__ == "__main__":
    df = cluster_users(n_clusters=4)
    result = dataframe_to_json(df)
    
    # Guarda como archivo JSON (opcional)
    with open("clusters_output.json", "w") as f:
        json.dump(result, f, indent=2)

    # Muestra en consola
    print(json.dumps(result[:5], indent=2))  # muestra primeros 5
