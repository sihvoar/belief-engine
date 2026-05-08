# Contributing to Bayes Tree

Thank you for considering contributing! This project benefits from community input — whether that's bug reports, feature suggestions, new example trees, or code improvements.

## Ways to Contribute

### 🐛 Report a Bug
- Open an [issue](https://github.com/sihvoar/belief-engine/issues) with steps to reproduce
- Include your YAML file if relevant
- Note your Python version and OS

### 💡 Suggest a Feature
- Open an issue with the `enhancement` label
- Describe the use case and expected behavior

### 📝 Add an Example Tree
- Create a well-structured YAML evidence tree for an interesting question
- Include a mix of `for`, `against`, and `neutral` evidence
- Submit a PR to the `examples/` directory

### 🔧 Submit Code
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run the validation suite: `python validation/run_validation.py`
5. Commit with a descriptive message
6. Push and open a Pull Request

## Development Setup

```bash
git clone https://github.com/sihvoar/belief-engine.git
cd belief-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
# Full validation suite (50 tests)
python validation/run_validation.py

# Quick CLI test
python -m bayes_tree.cli examples/napoleon.yaml --format json -n 1000
```

## Code Style

- Keep functions focused and well-documented
- Add type hints for new code
- Follow existing patterns in `bayes_engine.py`
- No external dependencies for the core engine (only `pyyaml`)

## Architecture

| File | Purpose |
|------|---------|
| `bayes_engine.py` | Core math and simulation (standalone, no GUI deps) |
| `bayes_tree/` | Pip-installable package (re-exports engine) |
| `bayes_tree/cli.py` | CLI with `--format json/csv` and `--prior-sweep` |
| `bayes-tree-eng.py` | Original terminal UI (colored output, histogram) |
| `bayes_tree_gui.py` | PyQt6 desktop GUI |
| `report_generator.py` | PDF report generation |
| `validation/` | 50-test validation suite |

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
