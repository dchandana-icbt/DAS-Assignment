from student_ml.train import train_all
from student_ml.predict import predict_all

if __name__ == "__main__":
    train_all("data/raw/student_data.csv", "models")
    predict_all("data/raw/student_data.csv", "models", output_csv="data/predictions/predictions.csv")
    print("Done. See data/predictions/predictions.csv")
