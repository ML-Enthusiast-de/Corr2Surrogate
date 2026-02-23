# Git Hooks

Enable repository hooks locally:

```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
```

The pre-commit hook runs:

```bash
PYTHONPATH=src python -m corr2surrogate.security.git_guard
```

If potential leaks are detected (API keys, tokens, user-home paths), the commit is blocked.
