# Recent commit failure analysis

## Scope
Investigated the two latest commits:
- `d5b5198` — `refactor: unify training, comparison, and dataset tools with deprecation wrappers`
- `63c8eb9` — `feat: add CSI source, enhance pipeline/preprocess, update training scripts`

## Finding 1: Python formatting gate is currently a hard failure
CI job `python-quality` runs `black --check apps/ tools/ tests/`.
Although the step name says "allow failure", the workflow does **not** append `|| true` and does not set `continue-on-error`, so any non-zero exit code fails the job.

Local reproduction on both commits:
- `black --check` exits with status `1`
- `63c8eb9`: reports `62 files would be reformatted`
- `d5b5198`: reports `65 files would be reformatted`

This means both commits can fail at the same quality gate even when functional code is valid.

## Finding 2: C++ configure path depends on external GTest download
`cpp-build-tests` runs `cmake --preset x86-debug`.
CMake declares GoogleTest via `FetchContent` from GitHub ZIP URL.
When that URL is unreachable or blocked, configuration fails before build/tests.

Local reproduction on both commits:
- `cmake --preset x86-debug` fails during `googletest-populate-download`
- Error observed: HTTP `403 Forbidden`

This introduces a second potential failure source independent of business logic changes.

## Suggested fixes
1. If black should be non-blocking, make it explicit in workflow:
   - add `continue-on-error: true` for that step, or
   - append `|| true`.
2. If formatting should be blocking, run `black apps/ tools/ tests/` once and commit the formatting baseline.
3. Avoid flaky GTest fetch by preferring one of:
   - install `libgtest-dev` in CI image and use system package,
   - vendor a pinned GTest tarball in internal mirror,
   - set `FETCH_GTEST=OFF` in restricted environments and use bundled `gtest-lite` fallback.
