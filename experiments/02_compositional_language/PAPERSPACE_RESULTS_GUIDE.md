# Guide: Handling Paperspace Training Results

## Option 1: Selective Git Commit (Recommended for Small Files)

### On Paperspace:
```bash
cd /notebooks/neural_networks_research/experiments/02_compositional_language

# Check file sizes first
du -sh output/*
du -sh outputs/*

# For files < 50MB, you can commit directly:
# 1. Add only essential files (avoid large model weights)
git add output/*.json output/*.txt output/*.md output/*.png
git add outputs/safeguarded_training/*.json
git add outputs/safeguarded_training/vocabulary.json

# 2. Commit with descriptive message
git commit -m "Add compositional language training results

Training completed successfully in 9 minutes
- Training history and metrics
- Evaluation results
- Vocabulary files
- Logs and summaries

Large model weights excluded - available on request"

# 3. Push to a results branch
git checkout -b compositional-results-$(date +%Y%m%d)
git push origin compositional-results-$(date +%Y%m%d)
```

### On Local Machine:
```bash
# Pull the results branch
git fetch origin
git checkout compositional-results-20250722  # Use actual date
git pull origin compositional-results-20250722
```

## Option 2: Create a Zip and Use GitHub Release (For Large Files)

### On Paperspace:
```bash
cd /notebooks/neural_networks_research/experiments/02_compositional_language

# Create a comprehensive results archive
zip -r compositional_results_$(date +%Y%m%d_%H%M%S).zip \
  output/ \
  outputs/safeguarded_training/ \
  -x "*.pyc" -x "__pycache__/*"

# Check size
ls -lh compositional_results_*.zip

# If < 2GB, can upload as GitHub release artifact
# If > 2GB, need to split or use cloud storage
```

Then create a GitHub release and attach the zip file.

## Option 3: Cloud Storage Transfer (For Very Large Results)

### On Paperspace:
```bash
# Use Paperspace's gradient storage or upload to cloud
# Example with Google Drive (if installed):
gdrive upload compositional_results_*.zip

# Or use wget-able service like transfer.sh:
curl --upload-file compositional_results_*.zip https://transfer.sh/
```

## What to Keep in Git vs External Storage

### Always Commit to Git:
- Training configuration files (`.json`)
- Training history/metrics (`.json`, `.csv`)
- Evaluation results (`.json`, `.txt`)
- Vocabulary files
- Small visualizations (`.png` < 1MB)
- Log files (`.log`, `.txt`)
- Summary reports (`.md`)

### Store Externally (GitHub Release/Cloud):
- Model weights (`.h5`, `.keras`, `.pt`)
- Large checkpoints
- Complete model directories
- Large datasets
- Video outputs

## Quick Decision Tree

1. **Total size < 50MB?** → Direct git commit
2. **Total size 50MB - 500MB?** → Selective commit (exclude weights)
3. **Total size 500MB - 2GB?** → GitHub Release
4. **Total size > 2GB?** → Cloud storage + link in README

## Preserving Results Long-term

Create a results README:

```bash
cat > output/README.md << EOF
# Compositional Language Training Results
Date: $(date)
Duration: 9 minutes
Machine: Paperspace P4000

## Files Included
- training_history.json: Loss and accuracy per epoch
- evaluation_results.json: Performance on test sets
- vocabulary.json: Tokenizer vocabulary
- final_model.h5: Trained model weights [Not in Git - too large]

## Key Metrics
[Add summary of results here]

## How to Load Model
\`\`\`python
from models import create_model
model = create_model(...)
model.load_weights('path/to/final_model.h5')
\`\`\`
EOF
```

## Recommended Approach for Your Case

Since training only took 9 minutes, the results are probably not huge. I recommend:

1. First check total size on Paperspace
2. If < 100MB total: Commit everything except .h5 files
3. Create a GitHub release for the complete results zip
4. Update the experiment's CURRENT_STATUS.md with results summary
