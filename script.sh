cat <<'EOF' > split_frontend_backend.sh
#!/usr/bin/env bash
set -euo pipefail

echo "=== Starting separation of frontend and backend ==="

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: not inside a git repository. Abort."
  exit 1
fi

mkdir -p frontend backend shared

echo "[1/7] Moving frontend-related files into ./frontend"
git mv pages frontend/ || true
git mv public frontend/ || true
git mv tsconfig.json frontend/ || true
git mv next.config.ts frontend/ || true
git mv postcss.config.mjs frontend/ || true
git mv eslint.config.mjs frontend/ || true
git mv package.json frontend/ || true
git mv package-lock.json frontend/ || true
git mv next-env.d.ts frontend/ || true

if [ -d ".next" ]; then
  echo "Note: .next directory exists. It's a build artifact; you may remove it or leave it (will be rebuilt)."
fi

echo "[2/7] Moving backend/ML-related files into ./backend"
git mv simple_model.py backend/ || true
git mv deploy_to_sagemaker.py backend/ || true
git mv test_simple_inference.py backend/ || true
git mv simple_inference backend/ || true
git mv ml backend/ || true
git mv model backend/ || true
git mv data backend/ || true
git mv bundle backend/ || true

echo "[3/7] Shared utilities"
if [ -d "shared" ]; then
  echo "shared/ already exists; leaving it."
else
  echo "No top-level shared/ found; skipping."
fi

echo "[4/7] Env and runtime artifacts"
if [ -f ".env.local" ]; then
  echo "Creating .env.example from .env.local (placeholder values)"
  awk -F= '{print $1"=REPLACE_ME"}' .env.local > .env.example
  git add .env.example
else
  echo "No .env.local; skipping .env.example creation."
fi

grep -qxF "response.json" .gitignore || echo "response.json" >> .gitignore
grep -qxF "payload.json" .gitignore || echo "payload.json" >> .gitignore
grep -qxF ".next/" .gitignore || echo ".next/" >> .gitignore
grep -qxF "frontend/node_modules/" .gitignore || echo "frontend/node_modules/" >> .gitignore
grep -qxF "backend/__pycache__/" .gitignore || echo "backend/__pycache__/" >> .gitignore
grep -qxF "*.pyc" .gitignore || echo "*.pyc" >> .gitignore

echo "[5/7] Staging changes"
git add frontend backend shared .gitignore

echo "[6/7] Writing reorg README stub"
cat <<'INNER' > REORG_README.md
# Repository reorganized: frontend / backend split

## Layout now

- frontend/  
  Contains the Next.js dashboard app.
- backend/  
  Contains Python/SageMaker model and deployment code.
- shared/  
  Shared utilities if applicable.

## Next steps

1. Fix import paths if broken.
2. Update CI/deploy scripts for new paths.
3. Run frontend from frontend/ and backend in its own env.
4. Use .env.example to populate .env.local securely.
INNER

git add REORG_README.md

echo "[7/7] Quick scan for stale references"
grep -R --line-number -E "pages/" -n . || true

echo "=== Separation script done. ==="
echo "Review with 'git status' and 'git diff', then commit:"
echo "  git commit -m 'Split repo into frontend/ and backend/ directories; added .env.example and cleanup stubs.'"
EOF
