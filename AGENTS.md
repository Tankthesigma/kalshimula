# Team Workflow

## Repository workflow

- Work in the same repository using short-lived feature branches.
- Open pull requests for review before merging to `main`.
- Keep changes focused so reviews are small and clear.
- Address review comments before merge.

## AI editing coordination

- Only one AI agent should edit the repository at a time.
- Before starting, confirm the current branch and scope of files to change.
- Do not overwrite or revert another contributor's work unless explicitly instructed.

## Safety rules

- Do not add auto-trading behavior.
- Do not add Kalshi API integration or trading logic unless explicitly approved in a dedicated task.
- Do not commit secrets, API keys, tokens, credentials, or private configuration.
- Keep secrets in local environment variables or an approved secret manager only.

## Before opening a PR

- Run `ruff check .`.
- Run `pytest`.
- Include the test results in the pull request description.
