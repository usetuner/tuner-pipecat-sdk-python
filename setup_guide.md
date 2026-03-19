Create a venv with **Python 3.12 or 3.13** (not the system 3.14 if that is your default):

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # macOS/Linux
# or: uv venv --python 3.12 && source .venv/bin/activate
```

**Into that venv:**

```bash
pip install -e .
# optional dev/test extras
pip install -e ".[dev]"
```

**Examples** under `examples/` depend on this SDK via a **local path** (`tool.uv.sources`). Use **`uv sync`** inside each example directory, or install the SDK from the repo root with `pip install -e .` first, then `pip install -e .` in the example (plain `pip` does not read `uv.sources`).


## Examples

See [`examples/`](examples/) for working bots built with this SDK:

| Example | Use case |
|---------|----------|
| [`pizza_order/`](examples/pizza_order/) | Pizza ordering with toppings, delivery/pickup branch |
| [`appointment_booking/`](examples/appointment_booking/) | Medical clinic receptionist, 7-node linear flow |
| [`customer_support/`](examples/customer_support/) | Multi-branch support agent with escalation |

Each example is self-contained. See **Installation** above for how the SDK is resolved. To run one:

```bash
cd examples/<example_name>
uv sync
cp .env.example .env   # if present; fill in API keys
uv run <script>.py
```

Then open http://localhost:7860 where applicable.

## Publish to Pypi
1. Install tools (one-time)

uv sync --extra dev

2. Build the package

cd src/pipecat_flows_tuner
hatch build


3. Validate before uploading

twine check dist/*

4. (Optional) Test on TestPyPI first

twine upload --repository testpypi dist/*

5. Upload to PyPI
twine upload dist/*
