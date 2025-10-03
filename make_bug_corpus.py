# make_bug_corpus.py
import os, json, glob
import argparse

def collect_json_to_jsonl(input_dir, output_file):
    files = glob.glob(os.path.join(input_dir, "**/*.json"), recursive=True)
    print(f"[info] found {len(files)} JSON files under {input_dir}")

    with open(output_file, "w", encoding="utf-8") as out:
        count = 0
        for f in files:
            try:
                with open(f, encoding="utf-8") as fin:
                    data = json.load(fin)
                    # 如果是单个对象
                    if isinstance(data, dict):
                        out.write(json.dumps(data, ensure_ascii=False) + "\n")
                        count += 1
                    # 如果是数组
                    elif isinstance(data, list):
                        for item in data:
                            out.write(json.dumps(item, ensure_ascii=False) + "\n")
                            count += 1
            except Exception as e:
                print(f"[warn] skip {f}: {e}")
        print(f"[ok] wrote {count} lines to {output_file}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="input folder (e.g. bugs_split)")
    ap.add_argument("--out", required=True, help="output jsonl file (e.g. bugs_corpus.jsonl)")
    args = ap.parse_args()

    collect_json_to_jsonl(args.indir, args.out)
