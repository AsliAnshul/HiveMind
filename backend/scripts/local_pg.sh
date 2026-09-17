#!/usr/bin/env bash
# A throwaway Postgres + pgvector for local development, with no root access.
#
# Debian/Ubuntu only. It downloads the distro's pgvector package, unpacks it
# into a local directory, and runs a private cluster on port 55432 using
# PostgreSQL 18's extension_control_path. Nothing is installed system-wide and
# nothing outside $PGROOT is touched.
#
#   ./scripts/local_pg.sh start     # start it and print the DATABASE_URL
#   ./scripts/local_pg.sh stop
#   ./scripts/local_pg.sh destroy   # stop and delete all data
#
# Requires: postgresql-18 (server binaries) and the distro's pgvector package
# available to `apt-get download`. For anything beyond development, point
# DATABASE_URL at Supabase instead.
set -euo pipefail

PGVER="${PGVER:-18}"
PGPORT="${PGPORT:-55432}"
PGROOT="${PGROOT:-$(cd "$(dirname "$0")/.." && pwd)/.localpg}"
PGDATA="$PGROOT/data"
PGBIN="/usr/lib/postgresql/$PGVER/bin"
VECTOR_ROOT="$PGROOT/pgvector"
DB_NAME="hivemind"
DB_USER="$(id -un)"

export PATH="$PGBIN:$PATH"

die() { echo "error: $*" >&2; exit 1; }

fetch_pgvector() {
  [ -d "$VECTOR_ROOT" ] && return
  echo "Fetching postgresql-$PGVER-pgvector..."
  mkdir -p "$PGROOT/deb"
  (cd "$PGROOT/deb" && apt-get download "postgresql-$PGVER-pgvector" >/dev/null) \
    || die "apt-get download failed. Install postgresql-$PGVER-pgvector system-wide instead."
  dpkg-deb -x "$PGROOT"/deb/*.deb "$VECTOR_ROOT"
}

start() {
  [ -x "$PGBIN/initdb" ] || die "PostgreSQL $PGVER binaries not found at $PGBIN"
  fetch_pgvector

  if [ ! -d "$PGDATA" ]; then
    echo "Initialising cluster in $PGDATA..."
    initdb -D "$PGDATA" -U "$DB_USER" --auth=trust >/dev/null
    {
      echo "port = $PGPORT"
      echo "listen_addresses = '127.0.0.1'"
      # Unix sockets are off: the socket path would exceed the 107-byte limit.
      echo "unix_socket_directories = ''"
      echo "extension_control_path = '\$system:$VECTOR_ROOT/usr/share/postgresql/$PGVER'"
      echo "dynamic_library_path = '\$libdir:$VECTOR_ROOT/usr/lib/postgresql/$PGVER/lib'"
    } >> "$PGDATA/postgresql.conf"
  fi

  pg_ctl -D "$PGDATA" -l "$PGROOT/postgres.log" start >/dev/null
  for _ in $(seq 1 20); do
    pg_isready -h 127.0.0.1 -p "$PGPORT" >/dev/null 2>&1 && break
    sleep 0.5
  done

  psql -h 127.0.0.1 -p "$PGPORT" -d postgres -tAc \
    "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" | grep -q 1 \
    || psql -h 127.0.0.1 -p "$PGPORT" -d postgres -q -c "CREATE DATABASE $DB_NAME"
  psql -h 127.0.0.1 -p "$PGPORT" -d "$DB_NAME" -q -c "CREATE EXTENSION IF NOT EXISTS vector"

  echo
  echo "Ready. pgvector $(psql -h 127.0.0.1 -p "$PGPORT" -d "$DB_NAME" -tAc \
    "SELECT extversion FROM pg_extension WHERE extname='vector'")"
  echo "export DATABASE_URL=\"postgresql://$DB_USER@127.0.0.1:$PGPORT/$DB_NAME\""
}

case "${1:-start}" in
  start)   start ;;
  stop)    pg_ctl -D "$PGDATA" stop >/dev/null && echo "Stopped." ;;
  destroy) pg_ctl -D "$PGDATA" stop >/dev/null 2>&1 || true; rm -rf "$PGROOT"; echo "Destroyed." ;;
  *)       die "usage: $0 {start|stop|destroy}" ;;
esac
