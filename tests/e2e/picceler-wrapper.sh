#!/bin/sh
# picceler currently resolves its runtime library (lib/libPiccelerRuntime.a)
# relative to its own process cwd rather than its executable location, so it
# only links correctly when invoked with cwd set to the build directory. lit
# runs each e2e test's RUN: lines with cwd set to the test's own directory
# (build/e2e/), so this wrapper cd's into the build directory before
# exec'ing picceler. The cd is scoped to this script's own child process, so
# it doesn't leak into subsequent RUN: lines, which still need cwd=build/e2e
# to resolve the img/ fixtures.
cd "$(dirname "$0")/.." || exit 1
exec ./picceler "$@"
