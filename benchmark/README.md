# Benchmark Legacy Evidence

`benchmark/` is frozen historical evidence.

Do not add new benchmark runs, JSON outputs, trace directories, or generated
artifacts here.

Use the product benchmark instead:

```bash
dflash benchmark --suite smoke --model <target>
```

New public benchmark outputs go under:

```text
.artifacts/dflash/benchmarks/
```

New trace and diagnostic outputs go under:

```text
.artifacts/dflash/traces/
.artifacts/dflash/diagnostics/
```

`benchmark/results/` remains in place for old comparisons only. Do not delete,
move, or append to it during normal development.
