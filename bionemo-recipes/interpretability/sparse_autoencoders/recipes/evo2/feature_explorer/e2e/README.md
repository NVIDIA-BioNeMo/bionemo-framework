# Dashboard e2e (Playwright) — no GPU, no model

The smoke (`dashboard.spec.js`) drives the dashboard against the **GPU-free mock backend**
(`../mock_backend.py`), so it needs **no Evo2 model, no SAE checkpoint, and no GPU** — just
Node, a browser, and a few small Python packages for the mock.

## Prereqs

- Node + npm
- `pip install fastapi uvicorn numpy` — for `mock_backend.py` (Playwright launches it as the test's web server)

## Run

```bash
cd bionemo-recipes/interpretability/sparse_autoencoders/recipes/evo2/feature_explorer    # the dashboard dir (this README's parent)
npm install
npx playwright install chromium     # one-time browser download
npm run test:e2e                    # boots mock_backend.py + Vite, runs the 4 smoke tests
```

`npm run test:e2e` starts both servers itself (see `playwright.config.js`) — you don't start them by hand.

## Watch / debug it

- `npx playwright test --ui` — GUI: run tests, step through, see a DOM snapshot at each step.
- `npx playwright test --headed` — watch it drive the dashboard in a visible browser.
- `npx playwright show-report` — HTML report (traces + screenshots) after a run.

To just *click around* the dashboard with no GPU (not the test), from `bionemo-recipes/interpretability/sparse_autoencoders/recipes/evo2/feature_explorer`:
run `python mock_backend.py` then `npm run dev`, and open the Vite URL.
