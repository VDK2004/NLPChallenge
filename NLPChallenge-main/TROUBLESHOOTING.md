# Troubleshooting Guide

This document contains solutions to common issues you might encounter while running this project.

## Common Issues

### 1. HuggingFace Hub Import Error

**Issue:**
```
ImportError: cannot import name 'cached_download' from 'huggingface_hub'
```

**Description:**
This error occurs because newer versions of huggingface-hub have deprecated the `cached_download` function, which is still required by some dependencies in this project.

**Solution:**

1. Uninstall the current huggingface-hub package:
```bash
pip uninstall huggingface-hub -y
```

2. Install the compatible version:
```bash
pip install huggingface-hub==0.16.4
```

**Technical Details:**
- The error is related to compatibility between huggingface-hub and sentence-transformers
- Version 0.16.4 is known to work with the current project dependencies
- This is a temporary solution until the dependent packages are updated to use newer huggingface-hub APIs

If you encounter any other issues, please check this guide or report them in the project's issue tracker.
