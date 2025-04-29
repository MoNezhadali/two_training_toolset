#!/usr/bin/env bash
echo "Running Bandit Scan"
bandit -r src/ ./*.py
excod=$?

if [ $excod -ne 0 ]; then
  echo "Bandit encountered issues"
  exit 2
fi

echo "Bandit Scan Complete. No issues found."

echo "Running pip-audit"
pip-audit -r $SCOPE/requirements.txt
excod=$?

if [ $excod -eq 1 ]; then
  echo "pip-audit found vulnerabilities"
  exit 2
elif [ $excod -eq 2 ]; then
  echo "pip-audit encountered an error"
  exit 2
fi

echo "pip-audit Complete. No issues found."
