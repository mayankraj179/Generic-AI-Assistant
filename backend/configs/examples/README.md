# Inactive example configs

Files here are **not loaded** — `load_all_assistant_configs` only scans the
top level of `configs/`. They are kept as ready-made starting points.

- `finance_assistant_openrouter.yaml` — finance assistant running entirely on
  free OpenRouter models (chat + embeddings, tool calling). Parked 2026-09-28
  when all active assistants were moved back to Gemini-only. To re-activate,
  move it up into `configs/`, set `OPENROUTER_API_KEY`, and check the free
  model slugs still exist (they disappear without notice). Its previously
  ingested chunks are still in the DB under assistant_id
  `finance_assistant_openrouter`, tagged `nvidia/nemotron-3-embed-1b:free`.
