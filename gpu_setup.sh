#!/bin/bash
set -e

pip install --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12 cuml-cu12 cuvs-cu12 cupy-cuda12x

python -c "import cupy; print('cupy device:', cupy.cuda.runtime.getDeviceProperties(0)['name'])"
python -c "import cuml; print('cuml version:', cuml.__version__)"
python -c "import cuvs; print('cuvs version:', cuvs.__version__)"