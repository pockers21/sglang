SHELL := /usr/bin/env bash
MODEL_PATH ?=

.PHONY: help check environment core offline serving complete

help:
	@printf '%s\n' \
	  'make check                       Run syntax and unit tests' \
	  'make environment MODEL_PATH=...  Validate the SGLang/CUDA environment' \
	  'make core MODEL_PATH=...         Run CUDA Graph A/B plus NSYS' \
	  'make offline MODEL_PATH=...      Run the offline shape sweep' \
	  'make serving MODEL_PATH=...      Run a real HTTP concurrency sweep' \
	  'make complete MODEL_PATH=...     Run the complete evidence pipeline'

check:
	bash scripts/check_lab.sh

environment:
	test -n "$(MODEL_PATH)"
	bash scripts/check_environment.sh

core:
	test -n "$(MODEL_PATH)"
	bash scripts/run_full_study.sh "$(MODEL_PATH)"

offline:
	test -n "$(MODEL_PATH)"
	bash scripts/run_offline_sweep.sh "$(MODEL_PATH)"

serving:
	test -n "$(MODEL_PATH)"
	bash scripts/run_serving_sweep.sh "$(MODEL_PATH)"

complete:
	test -n "$(MODEL_PATH)"
	bash scripts/run_complete_lab.sh "$(MODEL_PATH)"
