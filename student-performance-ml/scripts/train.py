import argparse
from student_ml.train import train_all

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-col", default="Final_Status")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    res = train_all(
        args.input,
        args.out,
        target_col=args.target_col,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("Saved metadata:", res.metadata_path)
    print("Pass/Fail model:", res.passfail_model_path)

if __name__ == "__main__":
    main()