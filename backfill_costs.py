"""Fill NULL costs in ledger.db from prices.json (re-run any time; only touches NULL rows)."""
import json, sqlite3, sys
db = sys.argv[1] if len(sys.argv) > 1 else "ledger.db"
d = json.load(open("prices.json")); c = sqlite3.connect(db)
for m, p in d["models"].items():
    c.execute("""update model_calls set cost_in=(input_fresh*?+cache_read*?+cache_write*?)/1e6,
        cost_out=output_tokens*?/1e6, price_table_version=? where model_id=? and cost_in is null""",
        (p["input"], p["cache_read"], p["cache_write"], p["output"], d["version"], m))
c.commit(); print(c.execute("select count(*) from model_calls where cost_in is null").fetchone()[0], "still unpriced")
