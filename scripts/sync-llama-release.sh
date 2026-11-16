#!/usr/bin/env bash
#
# Make this package follow the latest llama.cpp version release.
#
# llama.cpp publishes two kinds of release:
# nightly builds, tagged b<number> and marked as pre-releases,
# and version releases, tagged vX.Y.Z.
# A version release has no XCFramework asset.
# Its nightly-tag.txt asset names the nightly build for the same commit,
# and that nightly release has the llama-<nightly>-xcframework.zip asset.
#
# This script converges the repository on one state:
#
#   1. Package.swift names the XCFramework for the latest version release,
#      and records that version in a comment above the URL.
#   2. The tag (X+2).Y.Z exists on a commit with that Package.swift.
#
# It does whichever part is missing and does nothing when both hold.
# When it changes something, it commits to the current branch,
# tags the commit, and pushes both to the remote in one atomic push.
#
# The script fails with an error annotation when the state is wrong
# and will not fix itself.
# It exits with success and a warning annotation for temporary failures,
# such as rate limits and server errors, so that the next run can retry.
#
# Usage: scripts/sync-llama-release.sh [--dry-run]
#
#   --dry-run  Read the releases and report what would change,
#              including a diff of Package.swift.
#              Do not change files, commit, tag, or push.
#
# Environment:
#   GITHUB_TOKEN       Optional. Token for GitHub API requests.
#   LLAMA_CPP_API_URL  Optional. API URL of the llama.cpp repository.
#                      Default: https://api.github.com/repos/ggml-org/llama.cpp
#   SYNC_REMOTE        Optional. Remote to fetch from and push to. Default: origin
#   SYNC_BRANCH        Optional. Branch to push to. Default: main
#
# Requires: bash 3.2 or later, git, curl, jq, and shasum or sha256sum.

set -euo pipefail

readonly MAPPED_MAJOR_OFFSET=2
# Upstream uploads nightly-tag.txt in a later step of the same workflow
# that creates the version release,
# so a release can briefly exist without it.
readonly NIGHTLY_TAG_GRACE_SECONDS=3600

API_URL="${LLAMA_CPP_API_URL:-https://api.github.com/repos/ggml-org/llama.cpp}"
REMOTE="${SYNC_REMOTE:-origin}"
BRANCH="${SYNC_BRANCH:-main}"
DRY_RUN=false

usage() {
    awk '/^# Usage:/ { p = 1 } p { line = $0; sub(/^# ?/, "", line); print line } /^# Requires:/ { exit }' "$0"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=true ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

notice() { echo "::notice::$*"; }

# Exits with failure: the state is wrong and will not fix itself.
fail() {
    echo "::error::$*"
    exit 1
}

# Exits with success: the failure is temporary and the next run can retry.
skip() {
    echo "::warning::$*; skipping this run."
    exit 0
}

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sync-llama-release.XXXXXX")" ||
    fail "Could not create a temporary directory."
trap 'rm -rf "$WORK_DIR"' EXIT

for tool in git curl jq; do
    command -v "$tool" >/dev/null 2>&1 || fail "Required tool not found: $tool"
done

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d ' ' -f 1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d ' ' -f 1
    else
        fail "Required tool not found: sha256sum or shasum"
    fi
}

# Request a URL and write the response body to a file.
# Returns only on success. Calls skip for temporary failures
# and fail for failures that a retry will not fix.
#
# Usage: request <url> <output-file> [api]
#   Pass "api" to send GitHub API headers and the token.
request() {
    local url="$1" output="$2" kind="${3:-}" code
    local headers="$WORK_DIR/headers"
    local args=(--silent --show-error --location --output "$output"
        --dump-header "$headers" --write-out '%{http_code}')

    if [ "$kind" = "api" ]; then
        args+=(--header "Accept: application/vnd.github+json")
        if [ -n "${GITHUB_TOKEN:-}" ]; then
            args+=(--header "Authorization: Bearer $GITHUB_TOKEN")
        fi
    fi

    if ! code="$(curl "${args[@]}" "$url")"; then
        skip "Request failed: $url"
    fi

    case "$code" in
        2??) return 0 ;;
        403 | 429)
            # GitHub uses 403 and 429 for rate limits,
            # and 403 for permission errors too.
            if grep -qi '^x-ratelimit-remaining: *0' "$headers" ||
                grep -qi '^retry-after:' "$headers" ||
                grep -qi 'rate limit' "$output"; then
                skip "GitHub rate limit reached (HTTP $code) for $url"
            fi
            fail "Request denied (HTTP $code) for $url: $(jq -r '.message // empty' "$output" 2>/dev/null || true)"
            ;;
        5??) skip "Server error (HTTP $code) for $url" ;;
        *) fail "Unexpected response (HTTP $code) for $url" ;;
    esac
}

# Exit status 0 when version $1 is lower than version $2.
version_lt() {
    local a1 a2 a3 b1 b2 b3
    IFS=. read -r a1 a2 a3 <<<"$1"
    IFS=. read -r b1 b2 b3 <<<"$2"
    [ "$a1" -lt "$b1" ] && return 0
    [ "$a1" -gt "$b1" ] && return 1
    [ "$a2" -lt "$b2" ] && return 0
    [ "$a2" -gt "$b2" ] && return 1
    [ "$a3" -lt "$b3" ]
}

readonly SEMVER_PATTERN='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$'
# Upstream's make-release-desc.sh selects the nightly tag with
# (^|-)b[0-9]+(-[0-9a-f]{7})?$; tags from branches other than master
# have a branch prefix and a commit suffix.
readonly NIGHTLY_PATTERN='^([A-Za-z0-9._]+-)*b[0-9]+(-[0-9a-f]{7})?$'
readonly URL_PATTERN='^https://[A-Za-z0-9._~/%-]+$'
readonly CHECKSUM_PATTERN='^[0-9a-f]{64}$'

repo_root="$(git rev-parse --show-toplevel)" || fail "Not in a git repository."
cd "$repo_root"
[ -f Package.swift ] || fail "Package.swift not found in $repo_root"

# MARK: - Read the latest version release

request "$API_URL/releases/latest" "$WORK_DIR/latest.json" api

version_tag="$(jq -r '.tag_name // empty' "$WORK_DIR/latest.json")"
[ -n "$version_tag" ] || skip "The latest release response has no tag_name"

if ! [[ "$version_tag" =~ ^v([0-9]+\.[0-9]+\.[0-9]+)$ ]] ||
    ! [[ "${BASH_REMATCH[1]}" =~ $SEMVER_PATTERN ]]; then
    fail "The latest llama.cpp release is $version_tag, not a vX.Y.Z version release."
fi
upstream_version="${version_tag#v}"
IFS=. read -r up_major up_minor up_patch <<<"$upstream_version"
mapped_version="$((up_major + MAPPED_MAJOR_OFFSET)).$up_minor.$up_patch"
notice "Latest llama.cpp version release: $version_tag (llama.swift $mapped_version)"

nightly_tag_url="$(jq -r '.assets[]? | select(.name == "nightly-tag.txt") | .browser_download_url' \
    "$WORK_DIR/latest.json" | head -n 1)"
if [ -z "$nightly_tag_url" ]; then
    age="$(jq -r '(now - (.published_at | fromdateiso8601)) | floor' "$WORK_DIR/latest.json" 2>/dev/null || true)"
    if [[ "$age" =~ ^-?[0-9]+$ ]] && [ "$age" -lt "$NIGHTLY_TAG_GRACE_SECONDS" ]; then
        skip "$version_tag was published ${age}s ago and has no nightly-tag.txt asset yet"
    fi
    fail "$version_tag has no nightly-tag.txt asset, so its nightly build is unknown. Upstream omits this file when no nightly tag points at the release commit."
fi

request "$nightly_tag_url" "$WORK_DIR/nightly-tag.txt"
nightly_tag="$(tr -d '[:space:]' <"$WORK_DIR/nightly-tag.txt")"
if ! [[ "$nightly_tag" =~ $NIGHTLY_PATTERN ]]; then
    fail "nightly-tag.txt for $version_tag is malformed: '$(head -c 200 "$WORK_DIR/nightly-tag.txt")'"
fi
notice "Nightly build for $version_tag: $nightly_tag"

# MARK: - Read the nightly release's XCFramework

request "$API_URL/releases/tags/$nightly_tag" "$WORK_DIR/nightly.json" api

asset_name="llama-$nightly_tag-xcframework.zip"
asset_url="$(jq -r --arg name "$asset_name" \
    '.assets[]? | select(.name == $name) | .browser_download_url' "$WORK_DIR/nightly.json" | head -n 1)"
asset_digest="$(jq -r --arg name "$asset_name" \
    '.assets[]? | select(.name == $name) | .digest // empty' "$WORK_DIR/nightly.json" | head -n 1)"

if [ -z "$asset_url" ]; then
    fail "Nightly release $nightly_tag for $version_tag has no $asset_name asset. Assets: $(jq -r '[.assets[]?.name] | join(", ")' "$WORK_DIR/nightly.json")"
fi
[[ "$asset_url" =~ $URL_PATTERN ]] || fail "Unexpected URL for $asset_name: $asset_url"

case "$asset_digest" in
    sha256:*) asset_digest="${asset_digest#sha256:}" ;;
    "") ;;
    *) fail "Unexpected digest format for $asset_name: $asset_digest" ;;
esac
if [ -n "$asset_digest" ] && ! [[ "$asset_digest" =~ $CHECKSUM_PATTERN ]]; then
    fail "Unexpected digest for $asset_name: $asset_digest"
fi

# MARK: - Read the current state

if ! git fetch --quiet --tags "$REMOTE" "+refs/heads/$BRANCH:refs/remotes/$REMOTE/$BRANCH"; then
    skip "Could not fetch tags from $REMOTE"
fi

# Prints "<version> <xcframework url> <checksum>" for a Package.swift file,
# with "-" for each value that is absent.
# Returns failure when the file has more than one of any value.
package_state() {
    local file="$1" version url checksum
    version="$(sed -n 's|^[[:space:]]*// llama\.cpp v\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\) .*|\1|p' "$file")"
    url="$(sed -n 's|^[[:space:]]*"\(https://[^"]*-xcframework\.zip\)",[[:space:]]*$|\1|p' "$file")"
    checksum="$(sed -n 's|^[[:space:]]*checksum: "\([0-9a-f]*\)"[[:space:]]*$|\1|p' "$file")"
    if [ "$(printf '%s\n' "$url" | grep -c .)" -gt 1 ] ||
        [ "$(printf '%s\n' "$checksum" | grep -c .)" -gt 1 ] ||
        [ "$(printf '%s\n' "$version" | grep -c .)" -gt 1 ]; then
        return 1
    fi
    echo "${version:--} ${url:--} ${checksum:--}"
}

state="$(package_state Package.swift)" ||
    fail "Package.swift has more than one XCFramework URL, checksum, or version comment."
read -r current_version current_url current_checksum <<<"$state"
[ "$current_url" != "-" ] || fail "Could not find the XCFramework URL in Package.swift."
[ "$current_checksum" != "-" ] || fail "Could not find the XCFramework checksum in Package.swift."
if [ "$current_version" = "-" ]; then
    notice "Package.swift: no recorded llama.cpp version, $current_url"
else
    notice "Package.swift: llama.cpp v$current_version, $current_url"
fi

if [ "$current_version" != "-" ] && version_lt "$upstream_version" "$current_version"; then
    fail "Package.swift records llama.cpp v$current_version, which is newer than the latest release $version_tag."
fi

newest_tag="$(git tag --list | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' |
    sort -t . -k 1,1n -k 2,2n -k 3,3n | tail -n 1 || true)"
if [ -n "$newest_tag" ] && version_lt "$mapped_version" "$newest_tag"; then
    fail "$version_tag maps to $mapped_version, which is lower than the newest tag $newest_tag."
fi

# Changes go on top of the remote branch, not on another checkout.
remote_head="$(git rev-parse --quiet --verify "refs/remotes/$REMOTE/$BRANCH^{commit}" || true)"
if [ "$(git rev-parse HEAD)" != "$remote_head" ]; then
    if [ "$DRY_RUN" = true ]; then
        echo "::warning::HEAD is not $REMOTE/$BRANCH; a run without --dry-run would stop here."
    else
        fail "HEAD is not $REMOTE/$BRANCH. Check out $BRANCH at $REMOTE/$BRANCH before you run this script."
    fi
fi

# MARK: - Plan

package_ok=false
if [ "$current_version" = "$upstream_version" ] && [ "$current_url" = "$asset_url" ] &&
    { [ -z "$asset_digest" ] || [ "$current_checksum" = "$asset_digest" ]; }; then
    package_ok=true
fi

tag_ok=false
if git rev-parse --quiet --verify "refs/tags/$mapped_version^{commit}" >/dev/null; then
    git show "refs/tags/$mapped_version:Package.swift" >"$WORK_DIR/tagged-Package.swift" ||
        fail "Tag $mapped_version has no Package.swift."
    state="$(package_state "$WORK_DIR/tagged-Package.swift")" ||
        fail "Package.swift at tag $mapped_version has more than one XCFramework URL, checksum, or version comment."
    read -r tagged_version tagged_url tagged_checksum <<<"$state"
    if [ "$tagged_url" != "$asset_url" ] ||
        { [ -n "$asset_digest" ] && [ "$tagged_checksum" != "$asset_digest" ]; }; then
        fail "Tag $mapped_version exists but names $tagged_url (recorded llama.cpp version: $tagged_version), not $asset_url. Tags cannot be moved."
    fi
    tag_ok=true
fi

if [ "$package_ok" = true ] && [ "$tag_ok" = true ]; then
    notice "Up to date: Package.swift names $nightly_tag for $version_tag, and tag $mapped_version exists."
    exit 0
fi

# MARK: - Update Package.swift

new_package="$WORK_DIR/Package.swift"
if [ "$package_ok" = false ]; then
    # Download the asset and check its bytes, because SwiftPM checks the
    # checksum only when a client resolves the package.
    archive="$WORK_DIR/$asset_name"
    request "$asset_url" "$archive"
    checksum="$(sha256_of "$archive")"
    if [ -n "$asset_digest" ] && [ "$checksum" != "$asset_digest" ]; then
        fail "Checksum of the downloaded $asset_name is $checksum, but GitHub reports $asset_digest."
    fi
    [[ "$checksum" =~ $CHECKSUM_PATTERN ]] || fail "Could not compute the checksum of $asset_name."
    notice "Checksum of $asset_name: $checksum"

    # Replace the version comment, URL, and checksum of the binary target.
    awk -v comment="// llama.cpp $version_tag ($nightly_tag)" \
        -v url="$asset_url" -v checksum="$checksum" '
        /^[[:space:]]*\/\/ llama\.cpp v[0-9]/ { next }
        /^[[:space:]]*url:[[:space:]]*$/ && !done_comment {
            indent = $0; sub(/[^[:space:]].*$/, "", indent)
            print indent comment
            done_comment = 1
        }
        /^[[:space:]]*"https:\/\/[^"]*-xcframework\.zip",[[:space:]]*$/ {
            indent = $0; sub(/[^[:space:]].*$/, "", indent)
            print indent "\"" url "\","
            next
        }
        /^[[:space:]]*checksum: "[0-9a-f]*"[[:space:]]*$/ {
            indent = $0; sub(/[^[:space:]].*$/, "", indent)
            print indent "checksum: \"" checksum "\""
            next
        }
        { print }
    ' Package.swift >"$new_package" || fail "Could not rewrite Package.swift."

    state="$(package_state "$new_package")" ||
        fail "The rewritten Package.swift has more than one XCFramework URL, checksum, or version comment."
    read -r new_version new_url new_checksum <<<"$state"
    if [ "$new_version" != "$upstream_version" ] || [ "$new_url" != "$asset_url" ] ||
        [ "$new_checksum" != "$checksum" ]; then
        fail "The rewritten Package.swift does not name $asset_name for $version_tag. Check its format."
    fi

    echo "::group::Package.swift changes"
    diff -u Package.swift "$new_package" || true
    echo "::endgroup::"
fi

if [ "$DRY_RUN" = true ]; then
    [ "$package_ok" = false ] &&
        notice "Dry run: would update Package.swift to $nightly_tag for $version_tag and commit it to $BRANCH."
    [ "$tag_ok" = false ] &&
        notice "Dry run: would create tag $mapped_version and push it to $REMOTE."
    exit 0
fi

# MARK: - Commit, tag, and push

refspecs=()
if [ "$package_ok" = false ]; then
    cp "$new_package" Package.swift || fail "Could not write Package.swift."
    if command -v swift >/dev/null 2>&1; then
        swift package dump-package >/dev/null || fail "The updated Package.swift does not load."
    else
        echo "::warning::swift not found; did not check that Package.swift loads."
    fi
    git add Package.swift || fail "Could not stage Package.swift."
    git commit --quiet --message "Update llama.cpp to $version_tag ($nightly_tag)" ||
        fail "Could not commit Package.swift."
    refspecs+=("HEAD:refs/heads/$BRANCH")
    notice "Committed Package.swift for $version_tag ($nightly_tag)."
fi

if [ "$tag_ok" = false ]; then
    git tag "$mapped_version" HEAD || fail "Could not create tag $mapped_version."
    refspecs+=("refs/tags/$mapped_version")
    notice "Created tag $mapped_version."
fi

git push --atomic "$REMOTE" "${refspecs[@]}" || fail "Could not push ${refspecs[*]} to $REMOTE."
notice "Pushed ${refspecs[*]} to $REMOTE."
