#!/bin/bash
# Serve the papers site
# Usage: ./serve.sh [port]

PORT=${1:-8080}
DIR="$(dirname "$0")"

cd "$DIR" || exit 1

echo "Building site..."
/usr/local/bin/hugo --minify --destination /var/www/papers 2>&1 || exit 1

echo "Serving at http://localhost:$PORT/papers/"
exec python3 "$DIR/tinyhttpd" "$PORT" /var/www/papers
