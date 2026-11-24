# Security policy

## Reporting a vulnerability

Do not publish API keys, private datasets, or exploit details in a public issue. Contact the repository maintainers privately through the contact method configured in the GitHub repository.

## Important execution warning

OptMaster executes LLM-generated Python code. Docker execution is the intended isolation mode, but it is not a formal security guarantee. Do not run untrusted generated code on a machine containing sensitive data. Local execution mode is not a sandbox and should only be used for trusted code or controlled development.

Keep `GEMINI_API_KEY` in a local `.env` file or environment variable. Never commit `.env` or place secrets in issue reports.
