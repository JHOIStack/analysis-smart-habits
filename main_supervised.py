from analysis.supervised import classify_profiles
from utils.export import dataframe_to_json
import json

if __name__ == "__main__":
    df = classify_profiles()
    result = dataframe_to_json(df)

    with open("classification_output.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result[:5], indent=2))
