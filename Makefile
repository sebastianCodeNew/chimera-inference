CXX ?= g++
CXXFLAGS = -O3 -mavx2 -mfma -pthread -fPIC -shared -Wall -Wextra

SO_TARGET = src/libchimera_core.so
SRC_CPP = src/chimera_core.cpp

.PHONY: all build test lint format server evaluate clean

all: build

build: $(SO_TARGET)

$(SO_TARGET): $(SRC_CPP)
	@echo "Compiling AVX2 C++ GEMV Kernel..."
	$(CXX) $(CXXFLAGS) $(SRC_CPP) -o $(SO_TARGET)
	@echo "Built $(SO_TARGET) successfully."

test: build
	PYTHONPATH=. .venv/bin/pytest -v

lint:
	.venv/bin/ruff check src tests tools
	.venv/bin/ruff format --check src tests tools

format:
	.venv/bin/ruff format src tests tools

server: build
	PYTHONPATH=. .venv/bin/python tools/run_server.py --host 0.0.0.0 --port 8000 --device cuda

evaluate:
	PYTHONPATH=. .venv/bin/python tools/evaluate_models.py --preset balanced --format table

clean:
	rm -f $(SO_TARGET)
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ tools/__pycache__ benchmarks/__pycache__
	rm -rf .pytest_cache .ruff_cache
