from model.project_runner import build_and_save_model

if __name__ == "__main__":
    path = build_and_save_model()
    print(f"Saved final SVR model to: {path}")
