"""Read token diagnostics without installing torch/verl."""

import argparse
import gzip
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="step_000001.jsonl.gz")
    parser.add_argument("--sample", type=int, default=0, help="Zero-based line index within this step")
    parser.add_argument("--start", type=int, default=0, help="Zero-based token index; negative means from end")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--top", type=int, default=3, help="Number of saved candidates to display")
    args = parser.parse_args()
    with gzip.open(args.path, "rt", encoding="utf-8") as handle:
        row = next((json.loads(line) for i, line in enumerate(handle) if i == args.sample), None)
    if row is None:
        parser.error("Sample index is outside the file")
    print(f"step={row['step']} phase={row['phase']} uid={row['uid']} valid_tokens={len(row['tokens'])}")
    print("position\ttoken\tstudent_p\tteacher_p\tlogp_S-logp_T\tentropy\twatch_probabilities\ttop_candidates")

    def number(value):
        return "NA" if value is None else f"{value:.6g}"

    start = args.start if args.start >= 0 else max(0, len(row["tokens"]) + args.start)
    for token in row["tokens"][start : start + args.count]:
        watch = "; ".join(f"{t['token']}={number(t['p'])}" for t in token["student_watch"])
        top = "; ".join(f"{t['token']!r}={number(t['p'])}" for t in token["student_topk"][: args.top])
        print(
            "\t".join(
                [
                    str(token["position"]),
                    repr(token["token"]),
                    number(token["student_p"]),
                    number(token["teacher_p"]),
                    number(token["logp_student_minus_teacher"]),
                    number(token["student_entropy"]),
                    watch,
                    top,
                ]
            )
        )


if __name__ == "__main__":
    main()
