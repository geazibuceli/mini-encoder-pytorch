import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Compare experiment reports.")
    parser.add_argument("reports", nargs="+", help="JSON evaluation reports")
    args = parser.parse_args()
    rows = []
    for report_path in args.reports:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        rows.append({"report": report_path, "accuracy": report.get("accuracy"), "f1": report.get("f1"),
                     "evaluation_loss": report.get("evaluation_loss")})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__": main()
