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
#              With uncommitted changes, report the plan for HEAD
#              and warn that a run without --dry-run would stop.
#
# The script reads Package.swift from HEAD, the commit it tags,
# and a run without --dry-run requires no changes in the index or working tree.
#
# Environment:
#   GITHUB_TOKEN       Optional. Token for GitHub API requests.
#   LLAMA_CPP_API_URL  Optional. API URL of the llama.cpp repository.
#                      Default: https://api.github.com/repos/ggml-org/llama.cpp
#   LLAMA_CPP_DOWNLOAD_URL
#                      Optional. Base URL of llama.cpp release downloads.
#                      Release asset URLs must equal <base>/<tag>/<name>.
#                      Default: https://github.com/ggml-org/llama.cpp/releases/download
#   SYNC_REMOTE        Optional. Remote to fetch from and push to. Default: origin
#   SYNC_BRANCH        Optional. Branch to push to. Default: main
#   SYNC_RETRY_DELAY   Optional. Seconds before the first retry of a git
#                      transport failure; later retries wait longer. Default: 10
#
# Requires: bash 3.2 or later, git, curl, jq, and shasum or sha256sum.

set -euo pipefail

readonly MAPPED_MAJOR_OFFSET=2
# Upstream uploads nightly-tag.txt in a later step of the same workflow
# that creates the version release,
# so a release can briefly exist without it.
readonly NIGHTLY_TAG_GRACE_SECONDS=3600

# If the upstream repository moves, change both URLs together:
# release asset URLs must equal <DOWNLOAD_URL>/<tag>/<name>.
API_URL="${LLAMA_CPP_API_URL:-https://api.github.com/repos/ggml-org/llama.cpp}"
DOWNLOAD_URL="${LLAMA_CPP_DOWNLOAD_URL:-https://github.com/ggml-org/llama.cpp/releases/download}"
REMOTE="${SYNC_REMOTE:-origin}"
BRANCH="${SYNC_BRANCH:-main}"
RETRY_DELAY="${SYNC_RETRY_DELAY:-10}"
readonly GIT_ATTEMPTS=3
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

# Prints the SHA-256 checksum of a file. Returns failure when no tool can.
sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d ' ' -f 1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d ' ' -f 1
    else
        echo "Required tool not found: sha256sum or shasum" >&2
        return 1
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

    local status=0
    code="$(curl "${args[@]}" "$url")" || status=$?
    case "$status" in
        0) ;;
        # Name resolution, connection, timeout, TLS handshake, and
        # transfer errors can pass; other errors, such as a malformed URL,
        # an unsupported protocol, or a certificate problem, will not.
        5 | 6 | 7 | 16 | 18 | 28 | 35 | 52 | 55 | 56 | 92)
            skip "Request failed (curl exit $status): $url"
            ;;
        *) fail "Request failed (curl exit $status): $url" ;;
    esac

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

# Run a git command that talks to the remote,
# and write its standard output to $WORK_DIR/git-stdout.
# Retries transport failures, then fails.
# Fails at once for configuration problems and ref conflicts.
remote_git() {
    local attempt=1 err="$WORK_DIR/git-stderr" last
    while true; do
        if git "$@" >"$WORK_DIR/git-stdout" 2>"$err"; then
            return 0
        fi
        cat "$err" >&2
        last="$(grep -v '^[[:space:]]*$' "$err" | tail -n 1 || true)"
        if grep -qiE "would clobber|\[rejected\]|couldn't find remote ref|does not appear to be a git repository|repository .* not found|authentication failed|permission denied|invalid refspec|not a valid ref" "$err"; then
            fail "git $1 failed: $last"
        fi
        if ! grep -qiE "could not resolve host|failed to connect|couldn't connect|connection (timed out|refused|reset)|operation timed out|early eof|rpc failed|unexpected disconnect|the remote end hung up|returned error: (5[0-9][0-9]|429)|ssl|tls|empty reply|transfer closed" "$err"; then
            fail "git $1 failed: $last"
        fi
        if [ "$attempt" -ge "$GIT_ATTEMPTS" ]; then
            fail "git $1 failed after $attempt attempts: $last"
        fi
        echo "::warning::git $1 failed (attempt $attempt of $GIT_ATTEMPTS); retrying."
        sleep "$((RETRY_DELAY * attempt))"
        attempt=$((attempt + 1))
    done
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
readonly DOWNLOAD_URL_PATTERN='^https?://[A-Za-z0-9._~:/%-]*[A-Za-z0-9._~:%-]$'
readonly NIGHTLY_TAG_MAX_BYTES=256
readonly CHECKSUM_PATTERN='^[0-9a-f]{64}$'

repo_root="$(git rev-parse --show-toplevel)" || fail "Not in a git repository."
cd "$repo_root"
git cat-file -e HEAD:Package.swift 2>/dev/null || fail "Package.swift not found at HEAD in $repo_root"

# A tag names a commit, so the plan uses the committed Package.swift.
# Changes in the index or working tree would not be in that commit.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    if [ "$DRY_RUN" = true ]; then
        echo "::warning::The index or working tree has changes; this plan uses Package.swift at HEAD, and a run without --dry-run would stop here."
    else
        git status --short --untracked-files=no
        fail "The index or working tree has changes. Commit or discard them before you run this script."
    fi
fi

# MARK: - Read the latest version release

[[ "$DOWNLOAD_URL" =~ $DOWNLOAD_URL_PATTERN ]] ||
    fail "LLAMA_CPP_DOWNLOAD_URL is not a valid base URL: $DOWNLOAD_URL"

# Reads the one asset named $2 from the release JSON in file $1.
# Sets found_asset_url and found_asset_digest ("" when GitHub reports none).
# Returns 1 when the release has no such asset,
# and fails when it has more than one or the fields have the wrong types.
read_asset() {
    local json="$1" name="$2" count asset
    jq -e '(.assets | type) == "array"' "$json" >/dev/null 2>&1 ||
        fail "The release response in $(basename "$json") has no assets array."
    count="$(jq --arg name "$name" '[.assets[] | select(.name == $name)] | length' "$json")" ||
        fail "Could not read the assets in $(basename "$json")."
    case "$count" in
        0) return 1 ;;
        1) ;;
        *) fail "The release has $count assets named $name." ;;
    esac
    asset="$(jq -c --arg name "$name" '[.assets[] | select(.name == $name)][0]' "$json")" ||
        fail "Could not read asset $name."
    found_asset_url="$(jq -r '.browser_download_url | if type == "string" then . else error("not a string") end' <<<"$asset")" ||
        fail "Asset $name has no string browser_download_url."
    found_asset_digest="$(jq -r '.digest | if . == null then "" elif type == "string" then . else error("not a string") end' <<<"$asset")" ||
        fail "Asset $name has a digest that is not a string."
}

request "$API_URL/releases/latest" "$WORK_DIR/latest.json" api
jq -e 'type == "object"' "$WORK_DIR/latest.json" >/dev/null 2>&1 ||
    fail "The latest release response is not a JSON object."

version_tag="$(jq -r '.tag_name | strings' "$WORK_DIR/latest.json")"
[ -n "$version_tag" ] || fail "The latest release response has no tag_name."

if ! [[ "$version_tag" =~ ^v([0-9]+\.[0-9]+\.[0-9]+)$ ]] ||
    ! [[ "${BASH_REMATCH[1]}" =~ $SEMVER_PATTERN ]]; then
    fail "The latest llama.cpp release is $version_tag, not a vX.Y.Z version release."
fi
upstream_version="${version_tag#v}"
IFS=. read -r up_major up_minor up_patch <<<"$upstream_version"
mapped_version="$((up_major + MAPPED_MAJOR_OFFSET)).$up_minor.$up_patch"
notice "Latest llama.cpp version release: $version_tag (llama.swift $mapped_version)"

if ! read_asset "$WORK_DIR/latest.json" nightly-tag.txt; then
    age="$(jq -r '(now - (.published_at | fromdateiso8601)) | floor' "$WORK_DIR/latest.json" 2>/dev/null || true)"
    if [[ "$age" =~ ^-?[0-9]+$ ]] && [ "$age" -lt "$NIGHTLY_TAG_GRACE_SECONDS" ]; then
        skip "$version_tag was published ${age}s ago and has no nightly-tag.txt asset yet"
    fi
    fail "$version_tag has no nightly-tag.txt asset, so its nightly build is unknown. Upstream omits this file when no nightly tag points at the release commit."
fi
nightly_tag_url="$found_asset_url"
if [ "$nightly_tag_url" != "$DOWNLOAD_URL/$version_tag/nightly-tag.txt" ]; then
    fail "Unexpected URL for nightly-tag.txt of $version_tag: $nightly_tag_url"
fi

request "$nightly_tag_url" "$WORK_DIR/nightly-tag.txt"

# Accept one tag with an optional final LF or CRLF, and nothing else.
nightly_tag_file="$WORK_DIR/nightly-tag.txt"
nightly_tag_bytes="$(wc -c <"$nightly_tag_file" | tr -d ' ')" ||
    fail "Could not read nightly-tag.txt."
nightly_tag_shown="$(head -c "$NIGHTLY_TAG_MAX_BYTES" "$nightly_tag_file" | jq -Rs . 2>/dev/null || echo '(unprintable)')"
if [ "$nightly_tag_bytes" -gt "$NIGHTLY_TAG_MAX_BYTES" ] ||
    [ "$(LC_ALL=C tr -d '\000' <"$nightly_tag_file" | wc -c | tr -d ' ')" != "$nightly_tag_bytes" ]; then
    fail "nightly-tag.txt for $version_tag is malformed: $nightly_tag_shown"
fi
nightly_tag="$(cat "$nightly_tag_file"; printf x)"
nightly_tag="${nightly_tag%x}"
case "$nightly_tag" in
    *$'\r\n') nightly_tag="${nightly_tag%$'\r\n'}" ;;
    *$'\n') nightly_tag="${nightly_tag%$'\n'}" ;;
esac
if ! [[ "$nightly_tag" =~ $NIGHTLY_PATTERN ]]; then
    fail "nightly-tag.txt for $version_tag is malformed: $nightly_tag_shown"
fi
notice "Nightly build for $version_tag: $nightly_tag"

# MARK: - Read the nightly release's XCFramework

request "$API_URL/releases/tags/$nightly_tag" "$WORK_DIR/nightly.json" api
jq -e 'type == "object"' "$WORK_DIR/nightly.json" >/dev/null 2>&1 ||
    fail "The response for nightly release $nightly_tag is not a JSON object."
if [ "$(jq -r '.tag_name | strings' "$WORK_DIR/nightly.json")" != "$nightly_tag" ]; then
    fail "The response for nightly release $nightly_tag has a different tag_name."
fi

asset_name="llama-$nightly_tag-xcframework.zip"
if ! read_asset "$WORK_DIR/nightly.json" "$asset_name"; then
    fail "Nightly release $nightly_tag for $version_tag has no $asset_name asset. Assets: $(jq -r '[.assets[].name | strings] | join(", ")' "$WORK_DIR/nightly.json")"
fi
asset_url="$found_asset_url"
asset_digest="$found_asset_digest"
if [ "$asset_url" != "$DOWNLOAD_URL/$nightly_tag/$asset_name" ]; then
    fail "Unexpected URL for $asset_name: $asset_url"
fi

case "$asset_digest" in
    sha256:*) asset_digest="${asset_digest#sha256:}" ;;
    "") ;;
    *) fail "Unexpected digest format for $asset_name: $asset_digest" ;;
esac
if [ -n "$asset_digest" ] && ! [[ "$asset_digest" =~ $CHECKSUM_PATTERN ]]; then
    fail "Unexpected digest for $asset_name: $asset_digest"
fi

# MARK: - Read the current state

[[ "$RETRY_DELAY" =~ ^[0-9]+$ ]] || fail "SYNC_RETRY_DELAY must be a whole number of seconds."
git remote get-url "$REMOTE" >/dev/null 2>&1 || fail "Remote $REMOTE is not configured."
remote_git fetch --tags "$REMOTE" "+refs/heads/$BRANCH:refs/remotes/$REMOTE/$BRANCH"

# Prints "<version> <xcframework url> <checksum>" for a Package.swift file,
# with "-" for each value that is absent.
# Returns failure when the file has more than one of any value.
package_state() {
    local file="$1" version url checksum
    version="$(sed -n 's|^[[:space:]]*// llama\.cpp v\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\) .*|\1|p' "$file")"
    url="$(sed -n 's|^[[:space:]]*"\(https\{0,1\}://[^"]*-xcframework\.zip\)",[[:space:]]*$|\1|p' "$file")"
    checksum="$(sed -n 's|^[[:space:]]*checksum: "\([0-9a-f]*\)"[[:space:]]*$|\1|p' "$file")"
    if [ "$(printf '%s\n' "$url" | grep -c .)" -gt 1 ] ||
        [ "$(printf '%s\n' "$checksum" | grep -c .)" -gt 1 ] ||
        [ "$(printf '%s\n' "$version" | grep -c .)" -gt 1 ]; then
        return 1
    fi
    echo "${version:--} ${url:--} ${checksum:--}"
}

head_package="$WORK_DIR/HEAD-Package.swift"
git show HEAD:Package.swift >"$head_package" || fail "Could not read Package.swift at HEAD."
state="$(package_state "$head_package")" ||
    fail "Package.swift at HEAD has more than one XCFramework URL, checksum, or version comment."
read -r current_version current_url current_checksum <<<"$state"
[ "$current_url" != "-" ] || fail "Could not find the XCFramework URL in Package.swift at HEAD."
[ "$current_checksum" != "-" ] || fail "Could not find the XCFramework checksum in Package.swift at HEAD."
if [ "$current_version" = "-" ]; then
    notice "Package.swift at HEAD: no recorded llama.cpp version, $current_url"
else
    notice "Package.swift at HEAD: llama.cpp v$current_version, $current_url"
fi

if [ "$current_version" != "-" ] && version_lt "$upstream_version" "$current_version"; then
    fail "Package.swift records llama.cpp v$current_version, which is newer than the latest release $version_tag."
fi

# Read tags from the remote, because a local tag may be unpublished
# after a failed push.
remote_git ls-remote --tags --refs "$REMOTE"
cp "$WORK_DIR/git-stdout" "$WORK_DIR/remote-tags" || fail "Could not save the remote tags."
newest_tag="$(sed -n 's|^[0-9a-f]*[[:space:]]refs/tags/||p' "$WORK_DIR/remote-tags" |
    grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' | sort -t . -k 1,1n -k 2,2n -k 3,3n | tail -n 1 || true)"
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

# MARK: - Determine the expected checksum

archive="$WORK_DIR/$asset_name"
downloaded_checksum=""

# Downloads the asset once and checks its bytes against the API digest.
download_asset() {
    [ -z "$downloaded_checksum" ] || return 0
    request "$asset_url" "$archive"
    downloaded_checksum="$(sha256_of "$archive")" ||
        fail "Could not compute the checksum of $asset_name."
    [[ "$downloaded_checksum" =~ $CHECKSUM_PATTERN ]] ||
        fail "Could not compute the checksum of $asset_name."
    if [ -n "$asset_digest" ] && [ "$downloaded_checksum" != "$asset_digest" ]; then
        fail "Checksum of the downloaded $asset_name is $downloaded_checksum, but GitHub reports $asset_digest."
    fi
    notice "Checksum of $asset_name: $downloaded_checksum"
}

# Without an API digest, only the bytes can show
# whether a manifest's checksum is right.
if [ -n "$asset_digest" ]; then
    expected_checksum="$asset_digest"
else
    notice "GitHub reports no digest for $asset_name; downloading it to compute the checksum."
    download_asset
    expected_checksum="$downloaded_checksum"
fi

# MARK: - Plan

package_ok=false
if [ "$current_version" = "$upstream_version" ] && [ "$current_url" = "$asset_url" ] &&
    [ "$current_checksum" = "$expected_checksum" ]; then
    package_ok=true
fi

tag_ok=false
tag_local_only=false
remote_tag_sha="$(awk -v ref="refs/tags/$mapped_version" '$2 == ref { print $1 }' "$WORK_DIR/remote-tags")"
local_tag_sha="$(git rev-parse --quiet --verify "refs/tags/$mapped_version" || true)"
if [ -n "$remote_tag_sha" ]; then
    if [ "$local_tag_sha" != "$remote_tag_sha" ]; then
        fail "Tag $mapped_version on $REMOTE is $remote_tag_sha, but the local tag is ${local_tag_sha:-missing}."
    fi
    git show "refs/tags/$mapped_version:Package.swift" >"$WORK_DIR/tagged-Package.swift" ||
        fail "Tag $mapped_version has no Package.swift."
    state="$(package_state "$WORK_DIR/tagged-Package.swift")" ||
        fail "Package.swift at tag $mapped_version has more than one XCFramework URL, checksum, or version comment."
    read -r tagged_version tagged_url tagged_checksum <<<"$state"
    if [ "$tagged_url" != "$asset_url" ]; then
        fail "Tag $mapped_version exists but names $tagged_url (recorded llama.cpp version: $tagged_version), not $asset_url. Tags cannot be moved."
    fi
    if [ "$tagged_checksum" != "$expected_checksum" ]; then
        fail "Tag $mapped_version exists but its Package.swift has checksum $tagged_checksum, not $expected_checksum for $asset_name. Tags cannot be moved."
    fi
    tag_ok=true
elif [ -n "$local_tag_sha" ]; then
    # An earlier run in this checkout created the tag but did not push it.
    # Push it only if it is on HEAD and HEAD has the right Package.swift.
    if [ "$package_ok" = true ] &&
        [ "$(git rev-parse "refs/tags/$mapped_version^{commit}")" = "$(git rev-parse HEAD)" ]; then
        tag_local_only=true
    else
        fail "Tag $mapped_version exists locally but not on $REMOTE, and it is not on a HEAD with the right Package.swift. Delete it with 'git tag -d $mapped_version' and run again."
    fi
fi

if [ "$package_ok" = true ] && [ "$tag_ok" = true ]; then
    notice "Up to date: Package.swift names $nightly_tag for $version_tag, and tag $mapped_version exists."
    exit 0
fi

# MARK: - Update Package.swift

new_package="$WORK_DIR/Package.swift"
if [ "$package_ok" = false ]; then
    # Check the bytes before writing a checksum, because SwiftPM checks it
    # only when a client resolves the package.
    download_asset
    checksum="$downloaded_checksum"

    # Replace the version comment, URL, and checksum of the binary target.
    awk -v comment="// llama.cpp $version_tag ($nightly_tag)" \
        -v url="$asset_url" -v checksum="$checksum" '
        /^[[:space:]]*\/\/ llama\.cpp v[0-9]/ { next }
        /^[[:space:]]*url:[[:space:]]*$/ && !done_comment {
            indent = $0; sub(/[^[:space:]].*$/, "", indent)
            print indent comment
            done_comment = 1
        }
        /^[[:space:]]*"https?:\/\/[^"]*-xcframework\.zip",[[:space:]]*$/ {
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
    ' "$head_package" >"$new_package" || fail "Could not rewrite Package.swift."

    state="$(package_state "$new_package")" ||
        fail "The rewritten Package.swift has more than one XCFramework URL, checksum, or version comment."
    read -r new_version new_url new_checksum <<<"$state"
    if [ "$new_version" != "$upstream_version" ] || [ "$new_url" != "$asset_url" ] ||
        [ "$new_checksum" != "$checksum" ]; then
        fail "The rewritten Package.swift does not name $asset_name for $version_tag. Check its format."
    fi

    echo "::group::Package.swift changes"
    diff -u --label "a/Package.swift (HEAD)" --label "b/Package.swift" "$head_package" "$new_package" || true
    echo "::endgroup::"
fi

if [ "$DRY_RUN" = true ]; then
    [ "$package_ok" = false ] &&
        notice "Dry run: would update Package.swift to $nightly_tag for $version_tag and commit it to $BRANCH."
    if [ "$tag_local_only" = true ]; then
        notice "Dry run: would push the existing local tag $mapped_version to $REMOTE."
    elif [ "$tag_ok" = false ]; then
        notice "Dry run: would create tag $mapped_version and push it to $REMOTE."
    fi
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

if [ "$tag_local_only" = true ]; then
    refspecs+=("refs/tags/$mapped_version")
    notice "Pushing the existing local tag $mapped_version."
elif [ "$tag_ok" = false ]; then
    git tag "$mapped_version" HEAD || fail "Could not create tag $mapped_version."
    refspecs+=("refs/tags/$mapped_version")
    notice "Created tag $mapped_version."
fi

remote_git push --atomic "$REMOTE" "${refspecs[@]}"
notice "Pushed ${refspecs[*]} to $REMOTE."
