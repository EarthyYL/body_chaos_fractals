PY  := python
EXT := $(shell $(PY) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUT := systems/restricted/_fast$(EXT)

$(OUT): systems/restricted/fast.cpp
	c++ -O3 -march=native -Wall -std=c++17 -shared -fPIC $(shell $(PY) -m pybind11 --includes) $< -o $@
