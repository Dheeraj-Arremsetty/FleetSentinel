"""Extract tool-call traces from AgentDojo published runs into a flat table.

Each row = one agent episode (trace). Label semantics (verified against
agentdojo/task_suite/task_suite.py:run_task_with_pipeline):
  - attack_type in (None, "none")  -> benign episode (no injection present)
  - security == True               -> injection task achieved => compromised (rogue)
  - security == False              -> attacked but resisted => behaviorally benign
"""

import json
import glob
import os
import sys

import pandas as pd

RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "agentdojo_repo", "runs")

MODELS = [
    "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini-2024-07-18",
    "gemini-2.0-flash-001",
    "meta-llama_Llama-3.3-70B-Instruct",
]


def tool_sequence(messages):
    """Returns (names, calls) where calls = [{"f": name, "a": {key: str(value)}}]."""
    seq, calls = [], []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function")
            if isinstance(fn, dict):  # some formats nest {name, arguments}
                fn = fn.get("name")
            if fn:
                seq.append(fn)
                args = tc.get("args") or {}
                calls.append({"f": fn, "a": {k: str(v)[:400] for k, v in args.items()}})
    return seq, calls


def n_error_tool_results(messages):
    n = 0
    for m in messages:
        if m.get("role") == "tool" and m.get("error"):
            n += 1
    return n


def main():
    rows = []
    for model in MODELS:
        files = glob.glob(os.path.join(RUNS_DIR, model, "**", "*.json"), recursive=True)
        for f in files:
            try:
                d = json.load(open(f))
            except Exception as e:
                print(f"skip {f}: {e}", file=sys.stderr)
                continue
            msgs = d.get("messages", [])
            atk = d.get("attack_type")
            benign = atk is None or atk == "none"
            # Benign runs of injection-task prompts are the *attacker's goal run
            # as a user task*; exclude them from the benign pool to keep the
            # benign class purely "normal role behavior".
            user_task = d.get("user_task_id") or ""
            if benign and user_task.startswith("injection_task"):
                continue
            seq, calls = tool_sequence(msgs)
            rows.append(
                dict(
                    model=model,
                    suite=d.get("suite_name"),
                    user_task=user_task,
                    attack_type=None if benign else atk,
                    injection_task=d.get("injection_task_id"),
                    compromised=bool(d.get("security")) and not benign,
                    attacked=not benign,
                    n_messages=len(msgs),
                    n_tool_calls=len(seq),
                    n_distinct_tools=len(set(seq)),
                    n_tool_errors=n_error_tool_results(msgs),
                    tools=" ".join(seq),
                    calls_json=json.dumps(calls),
                    path=os.path.relpath(f, RUNS_DIR),
                )
            )
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(__file__), "..", "data", "traces.parquet")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df)} traces to {out}")
    print(df.groupby(["suite"]).agg(
        benign=("attacked", lambda s: (~s).sum()),
        resisted=("compromised", lambda s: 0),  # placeholder, refined below
    ))
    summary = df.assign(
        klass=df.apply(
            lambda r: "benign" if not r.attacked else ("compromised" if r.compromised else "resisted"),
            axis=1,
        )
    ).groupby(["suite", "klass"]).size().unstack(fill_value=0)
    print(summary)


if __name__ == "__main__":
    main()
