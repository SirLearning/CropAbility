# Agent instructions (CropAbility)

This file is the **portable entry point** for AI coding agents (Cursor, Codex, Claude Code, etc.).  
Cursor loads the same policies from [`.cursor/rules/`](.cursor/rules/).

## Language policy (required)

**All project documentation and code comments must be written in English.**

See [.cursor/rules/english-only.mdc](.cursor/rules/english-only.mdc) for full rules and exceptions (e.g. locale-specific document templates).

## Project rules (Cursor)

| Rule file | Scope |
|-----------|--------|
| [english-only.mdc](.cursor/rules/english-only.mdc) | Always apply — English docs & comments |
| [cropability-project.mdc](.cursor/rules/cropability-project.mdc) | Always apply — repo layout & workflow |

Non-Cursor agents: read both `.mdc` files above; treat their body (below the YAML frontmatter) as binding instructions.

## Repository overview

- **Python package**: `src/main/python/cropability/` — genomics, GPU kernels, CLI (`cropability` command)
- **Java integration**: `src/main/java/com/example/triton/` — TorchScript via PyTorch Java API
- **Tests**: `src/test/python/`, `src/test/java/`
- **Docs**: `docs/`, root `README.md`
- **Ignored**: `src/main/resources/private/` (see `.gitignore`)

## Build & test

```bash
pip install -e ".[gpu,dev]"
pytest src/test/python
mvn test
cropability info
```

## Conventions for agents

1. Use English for new comments, docstrings, docs, and CLI help.
2. Keep diffs focused; follow patterns in the nearest module.
3. Do not add secrets or private PDFs under version control.
4. Prefer `src/main/python/cropability/` for new Python code over legacy root `cropability/` copies.
