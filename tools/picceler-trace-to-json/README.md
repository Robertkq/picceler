## picceler-trace-to-json

A standalone Python 3 script (stdlib only — no dependencies to install) that converts the `.bin`
profiling trace written by a `--profile`-compiled picceler binary into Chrome Trace Event JSON,
loadable directly in [Perfetto](https://ui.perfetto.dev) or [chrome://tracing](chrome://tracing). See
[`docs/profiling.md`](../../docs/profiling.md) for the full format spec and workflow.

```bash
python3 picceler-trace-to-json.py picceler_profiling_trace.bin -o trace.json
```
