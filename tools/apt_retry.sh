#!/usr/bin/env bash

_apt_get() {
  if [ "$(id -u)" -eq 0 ]; then
    apt-get -o Acquire::Retries=3 "$@"
  else
    sudo apt-get -o Acquire::Retries=3 "$@"
  fi
}

_apt_get_retry() {
  for attempt in 1 2 3; do
    if _apt_get "$@"; then
      return 0
    fi
    if [ "$attempt" = "3" ]; then
      return 1
    fi
    sleep $((attempt * 15))
  done
}

apt_get_update() {
  _apt_get_retry update
}

apt_get_install() {
  _apt_get_retry install -y "$@"
}
