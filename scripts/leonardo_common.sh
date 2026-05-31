#!/usr/bin/env bash

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer; got '${value}'" >&2
    exit 2
  fi
}

require_min_int() {
  local name="$1"
  local value="$2"
  local minimum="$3"
  require_positive_int "${name}" "${value}"
  if (( 10#${value} < 10#${minimum} )); then
    echo "${name} must be at least ${minimum}; got '${value}'" >&2
    exit 2
  fi
}

require_max_int() {
  local name="$1"
  local value="$2"
  local maximum="$3"
  require_positive_int "${name}" "${value}"
  if (( 10#${value} > 10#${maximum} )); then
    echo "${name} must be at most ${maximum}; got '${value}'" >&2
    exit 2
  fi
}

require_choice() {
  local name="$1"
  local value="$2"
  shift 2
  local choice
  for choice in "$@"; do
    if [[ "${value}" == "${choice}" ]]; then
      return 0
    fi
  done
  echo "${name} must be one of: $*; got '${value}'" >&2
  exit 2
}
