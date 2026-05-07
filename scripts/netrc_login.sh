#!/usr/bin/env bash
#
# Authenticate to Certora's private PyPI (AWS CodeArtifact) by writing a
# token into ~/.netrc. After this runs, any HTTP client that respects
# ~/.netrc -- including uv -- can install from the private index without
# embedding credentials in URLs or environment variables.
#
# Requires AWS credentials to be configured in the environment, e.g. via
# `aws configure` locally, or `aws-actions/configure-aws-credentials` in
# GitHub Actions.
#
# Token TTL is 12 hours; re-run after expiry.
#
# Output format matches `certora-cloud codeartifacts login` so the two are
# interchangeable.

set -euo pipefail

DOMAIN="${CODEARTIFACT_DOMAIN:-certora}"
DOMAIN_OWNER="${CODEARTIFACT_DOMAIN_OWNER:-092457480553}"
REGION="${CODEARTIFACT_REGION:-us-west-2}"

HOST="${DOMAIN}-${DOMAIN_OWNER}.d.codeartifact.${REGION}.amazonaws.com"

TOKEN=$(aws codeartifact get-authorization-token \
    --domain "$DOMAIN" \
    --domain-owner "$DOMAIN_OWNER" \
    --region "$REGION" \
    --query authorizationToken \
    --output text)

NETRC="${NETRC:-$HOME/.netrc}"

if [[ -f "$NETRC" ]]; then
    TMP=$(mktemp)
    grep -v "$HOST" "$NETRC" > "$TMP" || true
    mv "$TMP" "$NETRC"
fi

umask 077
printf 'machine %s login aws password %s\n' "$HOST" "$TOKEN" >> "$NETRC"
chmod 600 "$NETRC"

echo "Wrote CodeArtifact credentials for $HOST to $NETRC"
