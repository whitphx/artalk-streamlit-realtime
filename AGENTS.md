# Repository Guidance

- Project-relative paths are fine when they refer to files owned by the repo,
  such as `assets/...`, `scripts/...`, or paths relative to the script/project
  root.
- Do not hardcode user- or machine-specific absolute local paths in application
  code, launcher scripts, package metadata, or default configuration.
- Use environment variables, package-manager configuration, documented examples,
  or caller-provided arguments for local absolute paths such as Python
  interpreters, external asset directories, package checkouts, model paths, and
  cache locations.
- Launcher scripts may derive defaults from portable relative paths or
  already-provided environment variables. They should not assume `/home/...`,
  `/Users/...`, workstation-specific worktrees, or environment names.
- Documentation may include absolute local paths only when clearly labeled as
  examples or machine-specific setup notes; runnable code and scripts must
  remain portable by default.
