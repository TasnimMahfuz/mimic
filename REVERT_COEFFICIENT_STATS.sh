#!/bin/bash
# Revert coefficient stats changes

echo "Reverting coefficient stats changes..."

# Revert backend
if [ -f "mimic-platform/backend/app/services/mimic_service.py.backup" ]; then
    cp mimic-platform/backend/app/services/mimic_service.py.backup mimic-platform/backend/app/services/mimic_service.py
    echo "✓ Backend reverted"
else
    echo "✗ Backend backup not found"
fi

# Revert frontend
if [ -f "mimic-platform/frontend/src/components/CoefficientStats.tsx.backup" ]; then
    cp mimic-platform/frontend/src/components/CoefficientStats.tsx.backup mimic-platform/frontend/src/components/CoefficientStats.tsx
    echo "✓ Frontend reverted"
else
    echo "✗ Frontend backup not found"
fi

echo "Done! Restart backend and frontend to apply changes."
