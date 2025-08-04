# Archiving Checklist

## Quick Summary
We're archiving ~150 outdated files to reduce clutter by 70% and improve project navigation.

## Step-by-Step Process

### ✅ 1. Review the Plan
- [ ] Read PROJECT_ARCHIVING_PLAN.md
- [ ] Check files to be archived make sense
- [ ] Confirm nothing critical is being archived

### ✅ 2. Run Dry Run
```bash
python archive_project.py --dry-run
```
- [ ] Review what will be archived
- [ ] Check for any files that shouldn't be moved
- [ ] Verify archive structure looks correct

### ✅ 3. Create Git Commit Point
```bash
git add -A
git commit --no-gpg-sign -m "Pre-archive checkpoint: saving current state before cleanup"
```

### ✅ 4. Run Actual Archive
```bash
python archive_project.py
```
- [ ] Confirm when prompted
- [ ] Watch for any errors
- [ ] Check ARCHIVE_LOG.json was created

### ✅ 5. Verify Archive
- [ ] Check archive/ directory structure
- [ ] Spot-check a few files moved correctly
- [ ] Verify checksums in ARCHIVE_LOG.json

### ✅ 6. Remove Original Files (Optional)
- [ ] Script will prompt to remove originals
- [ ] Only say "yes" if archive verification passed
- [ ] Can skip this step if unsure

### ✅ 7. Update Documentation
- [ ] Update DOCUMENTATION_INDEX.md
- [ ] Update any broken references
- [ ] Add note to CLAUDE.md about archive

### ✅ 8. Commit Archive
```bash
git add -A
git commit --no-gpg-sign -m "Archive old experiments and failed approaches

- Archived ~150 files from pre-OOD illusion discovery
- Moved failed PINN, TTA, and complex architecture experiments
- Reduced active directory clutter by ~70%
- All files preserved in archive/ with checksums"
```

## What Gets Archived

### 🗂️ Major Categories
1. **Failed PINN experiments** (conclusively didn't work)
2. **TTA adaptation attempts** (failed approach)
3. **Old training script variants** (superseded by templates)
4. **Complex architectures** (V1/V2 that achieved 0%)
5. **Old status documents** (outdated plans)

### ✨ What Stays Active
- Current research plans and documentation
- Working baseline models
- Successful experiments
- Active evaluation frameworks
- Research diary (all entries)
- Utils and core infrastructure

## Benefits
- 70% less clutter
- Find files faster
- Clear working vs historical code
- All work preserved for reference

## If Something Goes Wrong
1. Check ARCHIVE_LOG.json for what was moved
2. Files are copied, not moved initially
3. Git history preserves everything
4. Can manually restore from archive/

Ready to start? Begin with the dry run!
