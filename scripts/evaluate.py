from src.train_model import train_model


if __name__ == "__main__":
    outputs = train_model()

    print("\nEvaluation Summary")
    print(outputs["results_df"].to_string(index=False))

    print("\nSaved artifacts:")
    print("- metrics summary CSV")
    print("- test predictions CSV")
    print("- feature importance CSV")
    print("- metrics.json")