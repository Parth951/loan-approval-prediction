# Contributing Guide

Thanks for your interest in contributing! Please follow these steps.

## Setup

1. Fork the repo and clone your fork
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows PowerShell
   pip install -r requirements.txt
   ```

## Making Changes

- Create a feature branch: `git checkout -b feat/<short-description>`
- Keep code readable with clear names and small functions
- Update docs when behavior or interfaces change (`README.md`, `docs/`)

## Testing

- Run the Flask app locally to verify UI and prediction flow:
  ```bash
  python app.py
  ```
- For training changes, run and review metrics/outputs:
  ```bash
  python train_ensemble.py
  ```

## Submitting PRs

- Rebase your branch on `main` and resolve conflicts locally
- Write a clear PR description: what changed and why
- Reference related issues if any

## Large Files

- Model files (`*.pth`) are tracked via Git LFS; avoid committing non-essential large binaries
