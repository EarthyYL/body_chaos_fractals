PY  := python
EXT := $(shell $(PY) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUT := systems/restricted/_fast$(EXT)
INC := $(shell $(PY) -m pybind11 --includes)

all: $(OUT) compile_flags.txt

$(OUT): systems/restricted/fast.cpp
	c++ -O3 -march=native -Wall -std=c++17 -shared -fPIC $(INC) $< -o $@

# include paths for clangd
compile_flags.txt:
	printf '%s\n' -std=c++17 -Wall $(INC) > $@
