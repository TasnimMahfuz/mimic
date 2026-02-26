#!/bin/bash
echo "Reverting stats tabs to single page..."
cp mimic-platform/frontend/src/components/CoefficientStats.tsx.safe_backup mimic-platform/frontend/src/components/CoefficientStats.tsx
echo "✓ Reverted to single-page version! Refresh browser to see changes."
echo ""
echo "To restore tabbed version, run:"
echo "cp mimic-platform/frontend/src/components/CoefficientStats.tsx.tabbed_backup mimic-platform/frontend/src/components/CoefficientStats.tsx"
