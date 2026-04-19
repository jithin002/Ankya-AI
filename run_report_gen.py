"""
run_report_gen.py
Generates all report charts + patches report_graphs.html with fresh data.
UTF-8 safe for Windows terminals.
"""
import sys, io, os

# Force UTF-8 on all output streams
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stdin  = io.StringIO("n\n")    # auto-answer 'n' to the live CER prompt

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# ── Now import and run ────────────────────────────────────────────────────────
import generate_report_visuals as G

benchmark = G.load_benchmark(G.BENCHMARK_PATH)
records   = G.load_eval_records(G.EVAL_PATH)

if not benchmark and not records:
    print("[ERROR] No data found. Run run_benchmark.py first.")
    sys.exit(1)

G.print_summary(benchmark, records)

print("\n[Charts] Generating PNG charts...")
if benchmark:
    G.plot_latency(benchmark)
    G.plot_cer_wer(benchmark)
if records:
    G.plot_ai_vs_human(records)
    G.plot_component_scores(records)

print("\n[HTML] Updating report_graphs.html with fresh JS data...")
G.regenerate_html(benchmark, records)

print("\n[Complete] Done. Open report_graphs.html in browser.\n")
