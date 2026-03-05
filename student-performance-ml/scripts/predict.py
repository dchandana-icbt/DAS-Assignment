import argparse
from student_ml.predict import predict_all

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = predict_all(args.input, args.models, output_csv=args.out)
    print("Wrote:", args.out)
    print(df.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
