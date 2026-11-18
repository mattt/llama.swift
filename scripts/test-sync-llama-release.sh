#!/usr/bin/env bash
#
# Test sync-llama-release.sh without network access.
#
# A fixture server (sync-llama-release-fixtures.py) plays the GitHub API
# and the release downloads, and serves a small generated file
# in place of the XCFramework archive.
# Each case runs the script in a scratch clone of a synthetic repository
# whose remote is a local bare repository,
# then checks the exit status and an annotation line.
# A stub swift command takes the place of `swift package dump-package`,
# because SwiftPM accepts only https URLs for binary targets.
#
# Usage: scripts/test-sync-llama-release.sh
#
# Set KEEP_TEST_DIR=1 to keep the scratch directory.
# Requires: bash 3.2 or later, git, curl, jq, python3, and shasum or sha256sum.

set -u

SCRIPTS="$(cd "$(dirname "$0")" && pwd)" || exit 2
S="$SCRIPTS/sync-llama-release.sh"
T="$(mktemp -d "${TMPDIR:-/tmp}/test-sync-llama-release.XXXXXX")" || exit 2
SERVER=""
cleanup() {
    if [ -n "$SERVER" ]; then
        kill "$SERVER" 2>/dev/null
        wait "$SERVER" 2>/dev/null
    fi
    if [ -n "${KEEP_TEST_DIR:-}" ]; then
        echo "Kept $T"
    else
        rm -rf "$T"
    fi
}
trap cleanup EXIT

must() { "$@" || { echo "SETUP FAILED: $*" >&2; exit 2; }; }

# Isolate git from the user's configuration, such as commit signing and hooks.
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
export GIT_AUTHOR_NAME="Sync Test" GIT_AUTHOR_EMAIL=sync-test@example.invalid
export GIT_COMMITTER_NAME="Sync Test" GIT_COMMITTER_EMAIL=sync-test@example.invalid
unset GITHUB_TOKEN SYNC_REMOTE SYNC_BRANCH SYNC_RETRY_DELAY

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d ' ' -f 1
    else
        shasum -a 256 "$1" | cut -d ' ' -f 1
    fi
}

# MARK: - Fixtures

must mkdir "$T/bin"
cat >"$T/bin/swift" <<'EOF'
#!/bin/sh
echo "$*" >>"$SWIFT_STUB_LOG"
if [ -n "${SWIFT_STUB_FAIL:-}" ]; then
    echo "error: stub manifest failure" >&2
    exit 1
fi
EOF
must chmod +x "$T/bin/swift"
export PATH="$T/bin:$PATH" SWIFT_STUB_LOG="$T/swift.log"

ARCHIVE="$T/archive.zip"
printf 'Fixture XCFramework archive for llama.swift tests\n' >"$ARCHIVE" || exit 2
CHECKSUM="$(sha256 "$ARCHIVE")" || exit 2

python3 -I "$SCRIPTS/sync-llama-release-fixtures.py" "$ARCHIVE" "$T/port" &
SERVER=$!
i=0
until [ -s "$T/port" ]; do
    i=$((i + 1))
    if [ "$i" -gt 100 ] || ! kill -0 "$SERVER" 2>/dev/null; then
        echo "SETUP FAILED: the fixture server did not start" >&2
        exit 2
    fi
    sleep 0.1
done
FX="http://127.0.0.1:$(cat "$T/port")"

# Every run uses the fixture server unless a case overrides it.
export LLAMA_CPP_API_URL="$FX/ok" LLAMA_CPP_DOWNLOAD_URL="$FX/ok/download"
# A port with no listener, for connection failures.
CLOSED="127.0.0.1:1"

# fx <case>: prints the environment that selects a fixture case.
fx() { echo "LLAMA_CPP_API_URL=$FX/$1 LLAMA_CPP_DOWNLOAD_URL=$FX/${1%%+*}/download"; }

# manifest <version comment or ""> <url> <checksum> [<url label>]
manifest() {
    cat <<EOF
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "llama.swift",
    products: [
        .library(name: "LlamaSwift", targets: ["LlamaSwift"])
    ],
    targets: [
        .binaryTarget(
            name: "llama-cpp",
EOF
    [ -z "$1" ] || echo "            $1"
    cat <<EOF
            ${4:-url:}
                "$2",
            checksum: "$3"
        ),
        .target(name: "LlamaSwift", dependencies: ["llama-cpp"]),
    ]
)
EOF
}

old_url() { echo "https://github.com/ggml-org/llama.cpp/releases/download/b$1/llama-b$1-xcframework.zip"; }
NEW_URL="$FX/ok/download/b12359/llama-b12359-xcframework.zip"
ZEROS="0000000000000000000000000000000000000000000000000000000000000000"
manifest "// llama.cpp v1.0.0 (b12359)" "$NEW_URL" "$CHECKSUM" >"$T/new-Package.swift" || exit 2

# The seed repository has three nightly-era commits, each tagged 2.<build>.0,
# and one commit with the Package.swift for v1.0.0 and no tag.
SEED="$T/seed"
must git init -q "$SEED"
for build in 10539 10545 10549; do
    manifest "" "$(old_url "$build")" "$(printf '%064d' "$build")" >"$SEED/Package.swift" || exit 2
    must git -C "$SEED" add Package.swift
    must git -C "$SEED" commit -q -m "Update llama.cpp release to b$build"
    must git -C "$SEED" tag "2.$build.0"
done
OLD="$(git -C "$SEED" rev-parse HEAD)"
must cp "$T/new-Package.swift" "$SEED/Package.swift"
must git -C "$SEED" commit -q -am "Follow llama.cpp version releases"
NEW="$(git -C "$SEED" rev-parse HEAD)"

# clone <name> <seed commit>: makes $T/<name>.git with the commit as main
# and the seed's tags, and clones it to $T/<name>.
clone() {
    rm -rf "${T:?}/$1.git" "${T:?}/$1"
    must git init -q --bare "$T/$1.git"
    must git -C "$T/$1.git" symbolic-ref HEAD refs/heads/main
    must git -C "$SEED" push -q "$T/$1.git" "$2:refs/heads/main" 'refs/tags/*:refs/tags/*'
    must git clone -q "$T/$1.git" "$T/$1"
}

# commit_manifest <dir> <message> <manifest arguments...>: commits a
# Package.swift without pushing it.
commit_manifest() {
    local dir="$1" message="$2"
    shift 2
    manifest "$@" >"$dir/Package.swift" || exit 2
    must git -C "$dir" commit -q -am "$message"
}

# MARK: - Checks

PASSED=0 FAILED=0
pass() { PASSED=$((PASSED + 1)); printf 'PASS %s\n' "$1"; }
fail() { FAILED=$((FAILED + 1)); printf 'FAIL %s\n' "$1"; }

# expect <label> <exit> <pattern> <dir> [<name>=<value>...] <command...>
# Checks the exit status and that an annotation line matches the ERE pattern.
expect() {
    local label="$1" want="$2" pattern="$3" dir="$4" out rc annotations
    shift 4
    out="$(cd "$dir" && env "$@" 2>&1)"
    rc=$?
    annotations="$(printf '%s\n' "$out" | grep -E '^::(error|warning|notice)::' || true)"
    if [ "$rc" = "$want" ] && printf '%s\n' "$annotations" | grep -qE -- "$pattern"; then
        PASSED=$((PASSED + 1))
        printf 'PASS %-44s exit=%s  %s\n' "$label" "$rc" \
            "$(printf '%s\n' "$annotations" | grep -E -- "$pattern" | tail -n 1 | cut -c 1-160)"
    else
        FAILED=$((FAILED + 1))
        printf 'FAIL %-44s exit=%s (want %s), want /%s/\n' "$label" "$rc" "$want" "$pattern"
        printf '%s\n' "$out" | sed 's/^/     | /'
    fi
}

# check <label> <command...>: passes when the command succeeds.
check() {
    local label="$1"
    shift
    if "$@" >/dev/null 2>&1; then pass "$label"; else fail "$label"; fi
}

# remote_is <name> <commit> [<tag>...]: checks that $T/<name>.git has main
# at the commit and exactly the seed's tags plus the given tags.
remote_is() {
    local name="$1" commit="$2" want
    shift 2
    want="$( (git -C "$SEED" tag -l; printf '%s\n' "$@") | grep . | sort)"
    [ "$(git -C "$T/$name.git" rev-parse main)" = "$commit" ] &&
        [ "$(git -C "$T/$name.git" tag -l | sort)" = "$want" ]
}

UPD='would update Package.swift to b12359 for v1.0.0'
TAG='would create tag 3.0.0'
ERR='^::error::.*'

clone main-state "$OLD"
clone branch-state "$NEW"

echo "## Default fixture, scratch remotes"
expect "main state, dry run" 0 "$UPD" "$T/main-state" "$S" --dry-run
expect "main state, dry run plans the tag" 0 "$TAG" "$T/main-state" "$S" --dry-run
expect "branch state, dry run (tag only)" 0 "^::notice::Dry run: $TAG" "$T/branch-state" "$S" --dry-run
expect "branch state, real run" 0 'Pushed refs/tags/3.0.0 to origin' "$T/branch-state" "$S"
expect "branch state, rerun" 0 'Up to date' "$T/branch-state" "$S"
clone real-main "$OLD"
: >"$SWIFT_STUB_LOG"
expect "main state, real run" 0 'Pushed HEAD:refs/heads/main refs/tags/3.0.0' "$T/real-main" "$S"
check "main state, real run: Package.swift is byte-identical to the expected file" \
    cmp "$T/real-main/Package.swift" "$T/new-Package.swift"
check "main state, real run: checks that Package.swift loads" \
    grep -qx 'package dump-package' "$SWIFT_STUB_LOG"
check "main state, real run: remote has the commit and tag 3.0.0" \
    remote_is real-main "$(git -C "$T/real-main" rev-parse HEAD)" 3.0.0
expect "main state, rerun" 0 'Up to date' "$T/real-main" "$S"

echo
echo "## Uncommitted changes"
clone dirty-state "$OLD"
cp "$T/new-Package.swift" "$T/dirty-state/Package.swift"
expect "right working file over wrong HEAD, real run" 1 'index or working tree has changes' "$T/dirty-state" "$S"
expect "right working file over wrong HEAD, dry run" 0 'plan uses Package.swift at HEAD' "$T/dirty-state" "$S" --dry-run
expect "  dry run plans from HEAD" 0 "$UPD" "$T/dirty-state" "$S" --dry-run
git -C "$T/dirty-state" add Package.swift
expect "staged right file over wrong HEAD, real run" 1 'index or working tree has changes' "$T/dirty-state" "$S"
git -C "$T/dirty-state" tag 3.0.0 HEAD
expect "same with local tag on HEAD, real run" 1 'index or working tree has changes' "$T/dirty-state" "$S"
expect "same with local tag on HEAD, dry run" 1 'exists locally but not on origin' "$T/dirty-state" "$S" --dry-run
git -C "$T/dirty-state" reset -q --hard
expect "clean, local tag on wrong HEAD, real run" 1 'exists locally but not on origin' "$T/dirty-state" "$S"
check "uncommitted cases left the scratch remote unchanged" remote_is dirty-state "$OLD"

echo
echo "## Fixture cases, dry run from main state"
f() {
    local c="$1" rc="$2" pattern="$3"
    # shellcheck disable=SC2046 # fx prints words to split.
    expect "$c" "$rc" "$pattern" "$T/main-state" $(fx "$c") "$S" --dry-run
}
f ok 0 "$UPD"
f crlf-nightly-txt 0 "$UPD"
f no-newline-nightly-txt 0 "$UPD"
f prefixed-nightly-txt 0 'would update Package.swift to master-b12359-0123abc for v1.0.0'
f ok+no-digest 0 "$UPD"
f not-json 1 "$ERR"'The latest release response is not a JSON object'
f no-tag-name 1 "$ERR"'has no tag_name'
f no-assets 1 "$ERR"'has no assets array'
f not-semver 1 "$ERR"'is b12400, not a vX.Y.Z'
f prerelease-semver 1 "$ERR"'v1.1.0-rc1, not a vX.Y.Z'
f no-nightly-txt-old 1 "$ERR"'v1.0.0 has no nightly-tag.txt asset'
f no-nightly-txt-new 0 '^::warning::v1.0.0 was published .* no nightly-tag.txt asset yet'
f duplicate-nightly-txt 1 "$ERR"'The release has 2 assets named nightly-tag.txt'
f wrong-nightly-txt-url 1 "$ERR"'Unexpected URL for nightly-tag.txt'
f nightly-txt-url-not-string 1 "$ERR"'no string browser_download_url'
f multiline-nightly-txt 1 "$ERR"'nightly-tag.txt .* is malformed'
f extra-line-nightly-txt 1 "$ERR"'nightly-tag.txt .* is malformed'
f trailing-space-nightly-txt 1 "$ERR"'nightly-tag.txt .* is malformed'
f nul-nightly-txt 1 "$ERR"'nightly-tag.txt .* is malformed'
f long-nightly-txt 1 "$ERR"'nightly-tag.txt .* is malformed'
f malformed-nightly-txt 1 "$ERR"'nightly-tag.txt .* is malformed'
f empty-nightly-txt 1 "$ERR"'nightly-tag.txt .* is malformed'
f nightly-txt-404 1 "$ERR"'Unexpected response \(HTTP 404\)'
f nightly-not-json 1 "$ERR"'nightly release b12359 is not a JSON object'
f no-xcframework 1 "$ERR"'has no llama-b12359-xcframework.zip asset'
f duplicate-xcframework 1 "$ERR"'The release has 2 assets named llama-b12359-xcframework.zip'
f wrong-xcframework-url 1 "$ERR"'Unexpected URL for llama-b12359-xcframework.zip'
f digest-not-string 1 "$ERR"'digest that is not a string'
f digest-wrong-algorithm 1 "$ERR"'Unexpected digest format for .*: md5:'
f digest-not-hex 1 "$ERR"'Unexpected digest for .*: Z{64}'
f nightly-tag-name-mismatch 1 "$ERR"'has a different tag_name'
f nightly-release-404 1 "$ERR"'Unexpected response \(HTTP 404\)'
f digest-mismatch 1 "$ERR"'but GitHub reports 0{64}'
f download-503 0 '^::warning::Server error \(HTTP 503\) .*-xcframework.zip'
f older-than-newest 1 "$ERR"'maps to 2.9.0, which is lower than the newest tag 2.10549.0'
f rate-limit-403 0 '^::warning::GitHub rate limit reached \(HTTP 403\)'
f rate-limit-429 0 '^::warning::GitHub rate limit reached \(HTTP 429\)'
f forbidden-403 1 "$ERR"'Request denied \(HTTP 403\)'
f unauthorized-401 1 "$ERR"'Unexpected response \(HTTP 401\)'
f server-error-502 0 '^::warning::Server error \(HTTP 502\)'
f rate-limit-on-nightly 0 '^::warning::GitHub rate limit reached .*releases/tags/b12359'

echo
echo "## No digest: manifests with a right and a wrong checksum"
clone fx-state "$NEW"
m() {
    local label="$1" c="$2" rc="$3" pattern="$4"
    # shellcheck disable=SC2046 # fx prints words to split.
    expect "$label" "$rc" "$pattern" "$T/fx-state" $(fx "$c") "$S" --dry-run
}
m "digest, right manifest, no tag" ok 0 "^::notice::Dry run: $TAG"
m "no digest, right manifest, no tag" ok+no-digest 0 "^::notice::Dry run: $TAG"
m "  downloads to compute the checksum" ok+no-digest 0 'GitHub reports no digest'
must git -C "$T/fx-state" tag 3.0.0
must git -C "$T/fx-state" push -q origin 3.0.0
m "no digest, right manifest and tag" ok+no-digest 0 'Up to date'
must git -C "$T/fx-state" push -q origin :refs/tags/3.0.0
must git -C "$T/fx-state" tag -d 3.0.0 >/dev/null
commit_manifest "$T/fx-state" "Break checksum" "// llama.cpp v1.0.0 (b12359)" "$NEW_URL" "$ZEROS"
must git -C "$T/fx-state" push -q origin HEAD:main
m "no digest, wrong manifest, no tag" ok+no-digest 0 "$UPD"
must git -C "$T/fx-state" tag 3.0.0
must git -C "$T/fx-state" push -q origin 3.0.0
m "no digest, wrong manifest at tag" ok+no-digest 1 "$ERR"'Tag 3.0.0 exists but its Package.swift has checksum 0{64}'
m "digest, wrong manifest at tag" ok 1 "$ERR"'Tag 3.0.0 exists but its Package.swift has checksum 0{64}'

echo
echo "## Fetch and request failures"
git -C "$T/main-state" remote add dead "https://$CLOSED/x.git"
expect "missing remote" 1 "$ERR"'Remote nope is not configured' "$T/main-state" SYNC_REMOTE=nope "$S" --dry-run
expect "missing branch" 1 "$ERR""git fetch failed: .*couldn't find remote ref" "$T/main-state" SYNC_BRANCH=nope "$S" --dry-run
expect "unreachable remote (3 attempts)" 1 "$ERR"'git fetch failed after 3 attempts' "$T/main-state" SYNC_REMOTE=dead SYNC_RETRY_DELAY=0 "$S" --dry-run
git -C "$T/main-state" tag -f 2.10549.0 'HEAD~2' >/dev/null
expect "conflicting local tag" 1 "$ERR"'would clobber existing tag' "$T/main-state" "$S" --dry-run
must git -C "$T/main-state" fetch -q -f origin 'refs/tags/2.10549.0:refs/tags/2.10549.0'
expect "curl: unsupported protocol" 1 "$ERR"'Request failed \(curl exit 1\)' "$T/main-state" LLAMA_CPP_API_URL=htp://example.invalid/x "$S" --dry-run
expect "curl: connection refused" 0 '^::warning::Request failed \(curl exit 7\)' "$T/main-state" "LLAMA_CPP_API_URL=http://$CLOSED/x" "$S" --dry-run
expect "invalid download base URL" 1 "$ERR"'LLAMA_CPP_DOWNLOAD_URL is not a valid base URL' "$T/main-state" 'LLAMA_CPP_DOWNLOAD_URL=https://x/"' "$S" --dry-run

echo
echo "## Tags"
must git -C "$T/branch-state.git" tag -d 3.0.0 >/dev/null
expect "local-only tag on HEAD, dry run" 0 'would push the existing local tag 3.0.0' "$T/branch-state" "$S" --dry-run
expect "local-only tag on HEAD, real run" 0 'Pushed refs/tags/3.0.0 to origin' "$T/branch-state" "$S"
must git -C "$T/branch-state.git" tag -d 3.0.0 >/dev/null
must git -C "$T/branch-state" tag -f 3.0.0 'HEAD~1' >/dev/null
expect "local-only tag elsewhere" 1 "$ERR"'Tag 3.0.0 exists locally but not on origin' "$T/branch-state" "$S" --dry-run
must git -C "$T/branch-state" tag -d 3.0.0 >/dev/null
must git -C "$T/main-state" tag 4.0.0
must git -C "$T/main-state" push -q origin 4.0.0
expect "newer 4.0.0 tag on remote" 1 "$ERR"'maps to 3.0.0, which is lower than the newest tag 4.0.0' "$T/main-state" "$S" --dry-run
must git -C "$T/main-state" push -q origin :refs/tags/4.0.0
must git -C "$T/main-state" tag -d 4.0.0 >/dev/null
must git -C "$T/main-state" tag 3.0.0
must git -C "$T/main-state" push -q origin 3.0.0
expect "3.0.0 on a b10549 commit" 1 "$ERR"'Tag 3.0.0 exists but names .*b10549.*Tags cannot be moved' "$T/main-state" "$S" --dry-run
must git -C "$T/main-state" push -q origin :refs/tags/3.0.0
must git -C "$T/main-state" tag -d 3.0.0 >/dev/null
# A tag on a commit with no Package.swift.
empty_tree="$(git -C "$T/main-state" hash-object -t tree /dev/null)" || exit 2
orphan="$(git -C "$T/main-state" commit-tree -m Orphan "$empty_tree")" || exit 2
must git -C "$T/main-state" tag 3.0.0 "$orphan"
must git -C "$T/main-state" push -q origin 3.0.0
expect "3.0.0 on a commit with no Package.swift" 1 "$ERR"'Tag 3.0.0 has no Package.swift' "$T/main-state" "$S" --dry-run
must git -C "$T/main-state" push -q origin :refs/tags/3.0.0
must git -C "$T/main-state" tag -d 3.0.0 >/dev/null
must git -C "$T/main-state" checkout -q -b other
must git -C "$T/main-state" commit -q --allow-empty -m other
expect "HEAD is not origin/main, real run" 1 "$ERR"'HEAD is not origin/main' "$T/main-state" "$S"
expect "HEAD is not origin/main, dry run" 0 '^::warning::HEAD is not origin/main' "$T/main-state" "$S" --dry-run

echo
echo "## Package.swift at HEAD"
must mkdir "$T/not-a-repo"
expect "not in a git repository" 1 "$ERR"'Not in a git repository' "$T/not-a-repo" "GIT_CEILING_DIRECTORIES=$T" "$S" --dry-run
must git init -q "$T/no-manifest"
must git -C "$T/no-manifest" commit -q --allow-empty -m Empty
expect "no Package.swift at HEAD" 1 "$ERR"'Package.swift not found at HEAD' "$T/no-manifest" "$S" --dry-run
# shellcheck disable=SC2046 # fx prints words to split.
expect "manifest newer than the latest release" 1 "$ERR"'records llama.cpp v1.0.0, which is newer than the latest release v0.9.0' \
    "$T/fx-state" $(fx older-than-newest) "$S" --dry-run
clone manifest-state "$OLD"
commit_manifest "$T/manifest-state" "Remove URL" "" "" "$ZEROS"
expect "no XCFramework URL" 1 "$ERR"'Could not find the XCFramework URL' "$T/manifest-state" "$S" --dry-run
commit_manifest "$T/manifest-state" "Remove checksum" "" "$NEW_URL" ""
expect "no XCFramework checksum" 1 "$ERR"'Could not find the XCFramework checksum' "$T/manifest-state" "$S" --dry-run
commit_manifest "$T/manifest-state" "Add second URL" "" "$NEW_URL\",
                \"$NEW_URL" "$ZEROS"
expect "two XCFramework URLs" 1 "$ERR"'Package.swift at HEAD has more than one XCFramework URL' "$T/manifest-state" "$S" --dry-run
commit_manifest "$T/manifest-state" "Comment URL label" "" "$(old_url 10549)" "$ZEROS" "url:  // XCFramework"
expect "unexpected format" 1 "$ERR"'rewritten Package.swift does not name .* Check its format' "$T/manifest-state" "$S" --dry-run

echo
echo "## Arguments, manifest check, and push"
# shellcheck disable=SC2016 # The inner shell expands $1.
check "--help prints the usage" sh -c '"$1" --help | grep -q "^Usage: scripts/sync-llama-release.sh"' sh "$S"
exits_2() { "$@"; [ "$?" = 2 ]; }
check "unknown argument exits with status 2" exits_2 "$S" --nope
clone swift-fails "$OLD"
expect "updated Package.swift does not load" 1 "$ERR"'The updated Package.swift does not load' "$T/swift-fails" SWIFT_STUB_FAIL=1 "$S"
check "  remote unchanged" remote_is swift-fails "$OLD"
clone push-declined "$OLD"
printf '#!/bin/sh\necho "declined by test hook"\nexit 1\n' >"$T/push-declined.git/hooks/pre-receive" || exit 2
must chmod +x "$T/push-declined.git/hooks/pre-receive"
expect "push declined by the remote" 1 "$ERR"'git push failed: .*failed to push some refs' "$T/push-declined" "$S"
check "  remote unchanged" remote_is push-declined "$OLD"

echo
echo "Passed: $PASSED  Failed: $FAILED"
[ "$FAILED" -eq 0 ]
