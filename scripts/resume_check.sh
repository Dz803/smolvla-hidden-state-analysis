#!/usr/bin/env bash
set -u

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
active_file="$project_root/.planning/.active_plan"
workstation_python="$project_root/local/lingbot-conda-env-archived/bin/python"
benchmark_run="$project_root/archive/full_experiment/runs/benchmark_400_20260723T021424Z_e0638bea"
full_check=false
failures=0

if [[ "${1:-}" == "--full" ]]; then
  full_check=true
fi

check_file() {
  if [[ -e "$1" ]]; then
    printf 'OK      %s\n' "$1"
  else
    printf 'MISSING %s\n' "$1"
    failures=$((failures + 1))
  fi
}

printf 'project_root=%s\n' "$project_root"
printf 'git_head=%s\n' "$(git -C "$project_root" rev-parse --short HEAD 2>/dev/null || printf unknown)"
git -C "$project_root" status --short --branch 2>/dev/null || true

check_file "$project_root/README.md"
check_file "$project_root/AGENTS.md"
check_file "$project_root/docs/resume.md"
check_file "$project_root/docs/experiment_log.md"
check_file "$active_file"

if [[ -f "$active_file" ]]; then
  active_plan=$(tr -d '[:space:]' < "$active_file")
  printf 'active_plan=%s\n' "$active_plan"
  check_file "$project_root/.planning/$active_plan/task_plan.md"
  check_file "$project_root/.planning/$active_plan/findings.md"
  check_file "$project_root/.planning/$active_plan/progress.md"
fi

if [[ -x "$workstation_python" ]]; then
  printf 'OK      workstation Python available\n'
  PYTHONPATH="$project_root/src" "$workstation_python" -c 'import numpy, pandas, sklearn, smolvla_analysis; print("OK      core Python imports")' || failures=$((failures + 1))
else
  printf 'INFO    code-only checkout: workstation Python is not present\n'
fi

if [[ -d "$benchmark_run" ]]; then
  printf 'OK      canonical benchmark available\n'
else
  printf 'INFO    code-only checkout: canonical benchmark is not present\n'
fi

if [[ "$full_check" == true && -x "$workstation_python" ]]; then
  printf 'running targeted tests...\n'
  PYTHONPATH="$project_root/src" "$workstation_python" -m pytest -q "$project_root/tests" || failures=$((failures + 1))
fi

if (( failures > 0 )); then
  printf 'resume_check=FAIL failures=%d\n' "$failures"
  exit 1
fi

printf 'resume_check=PASS\n'
