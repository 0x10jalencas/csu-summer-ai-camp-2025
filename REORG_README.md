# Repository Reorganized: Frontend / Backend Split

## New Layout

The repository has been successfully reorganized into a clear frontend/backend separation:

### `frontend/`
Contains the Next.js dashboard application with React components.

**Structure:**
- `src/` - Next.js app source code
  - `app/` - App router pages and API routes
  - `lib/` - Utility functions and configurations
- `public/` - Static assets (images, fonts, etc.)
- `pages/` - Legacy pages router (if needed)
- Configuration files: `package.json`, `next.config.ts`, `tsconfig.json`, `postcss.config.mjs`, `eslint.config.mjs`, `next-env.d.ts`

### `backend/`
Contains Python/SageMaker model and deployment code.

**Structure:**
- `ml/` - Machine learning training and preprocessing scripts
- `simple_inference/` - Simple inference code and utilities
- `bundle/` - Model bundling and packaging code
- `model/` - Model configuration and artifacts
- `data/` - Training data and datasets
- Python scripts: `simple_model.py`, `deploy_to_sagemaker.py`, `test_simple_inference.py`
- `requirements.txt` - Python dependencies

### `shared/`
Shared utilities and schemas used by both frontend and backend.

**Structure:**
- `feature_schema.json` - Shared feature schema for ML model encoding
- `model/` - Shared model artifacts (if any)

## Usage Instructions

### Frontend Development
```bash
cd frontend
npm install
npm run dev
```

### Backend Development
```bash
cd backend
pip install -r requirements.txt
python simple_model.py
```

## Important Notes

1. **Feature Schema**: The `feature_schema.json` is now shared between frontend and backend. The frontend imports it from `../../../shared/feature_schema.json`.

2. **Import Paths**: Frontend imports have been updated to reference the shared schema properly.

3. **Git History**: All file moves were done using `git mv` to preserve git history where possible.

4. **Build Artifacts**: The `.gitignore` has been updated to handle build artifacts in their new locations:
   - `frontend/node_modules/`
   - `frontend/.next/`
   - `backend/__pycache__/`

## Next Steps

1. ✅ **File Organization** - Complete
2. ✅ **Import Path Updates** - Complete  
3. ✅ **GitIgnore Updates** - Complete
4. 🔄 **Test Functionality** - Verify imports and functionality work
5. 📝 **Update CI/CD** - Update deployment scripts for new paths (if applicable)
6. 📝 **Update Documentation** - Update any remaining references to old paths

## Verification Checklist

- [ ] Frontend builds successfully (`cd frontend && npm run build`)
- [ ] Backend scripts run without import errors
- [ ] Feature encoding works correctly with shared schema
- [ ] API routes function properly
- [ ] All imports resolve correctly

This reorganization maintains full functionality while providing a clean separation of concerns between frontend and backend components.