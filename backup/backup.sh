#!/bin/zsh
# Usage: copy backup.sh into any project, then:
# chmod +x backup.sh && ./backup.sh
# OR
# zsh backup.sh

set -euo pipefail

export PATH="/Users/arj/miniconda3/bin:/Users/arj/miniconda3/condabin:/opt/homebrew/bin:/opt/homebrew/sbin:/Library/Frameworks/Python.framework/Versions/3.13/bin:/usr/local/bin:/System/Cryptexes/App/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin"

RM_BIN="/bin/rm"
CP_BIN="/bin/cp"
MV_BIN="/bin/mv"
MKDIR_BIN="/bin/mkdir"
DITTO_BIN="/usr/bin/ditto"
FIND_BIN="/usr/bin/find"
XATTR_BIN="/usr/bin/xattr"
MKTEMP_BIN="/usr/bin/mktemp"
DATE_BIN="/bin/date"
WC_BIN="/usr/bin/wc"
TR_BIN="/usr/bin/tr"
AWK_BIN="/usr/bin/awk"
DIRNAME_BIN="/usr/bin/dirname"
BASENAME_BIN="/usr/bin/basename"
GIT_BIN="$(command -v git)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Detect project root: use git toplevel if available, else use script's directory
if [[ -n "$GIT_BIN" ]] && "$GIT_BIN" -C "$SCRIPT_DIR" rev-parse --show-toplevel &>/dev/null; then
  PROJECT_DIR="$("$GIT_BIN" -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
  IS_GIT_REPO=true
else
  PROJECT_DIR="$SCRIPT_DIR"
  IS_GIT_REPO=false
fi

PROJECT_NAME="$("$BASENAME_BIN" "$PROJECT_DIR")"
SOURCE_DIR="$PROJECT_DIR"
BACKUP_DIR="${PROJECT_DIR}/backup"
GITIGNORE_FILE="${SOURCE_DIR}/.gitignore"

DATE_TAG="$("$DATE_BIN" '+%b %-d, %Y')"
BASE_NAME="${PROJECT_NAME}-${DATE_TAG}"
FINAL_NAME="${BASE_NAME}"
ZIP_NAME="${FINAL_NAME}.zip"
ZIP_PATH="${BACKUP_DIR}/${ZIP_NAME}"

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURE HERE — folders/files to include even if gitignored
# Paths are relative to the project root. Add one per line, e.g.:
#   "data"
#   "output"
#   "models/weights"
# ══════════════════════════════════════════════════════════════════════════════
EXTRA_INCLUDE=(
  "data"
  # "output"
)

# ── Validation ────────────────────────────────────────────────────────────────

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Error: source directory not found: $SOURCE_DIR"
  exit 1
fi

if [[ "$IS_GIT_REPO" == false ]]; then
  echo "Warning: not a git repository — all files in $SOURCE_DIR will be included"
fi

if [[ "$IS_GIT_REPO" == true ]] && [[ ! -f "$GITIGNORE_FILE" ]]; then
  echo "Warning: .gitignore not found at $GITIGNORE_FILE"
fi

# ── Ensure backup directory exists ───────────────────────────────────────────

"$MKDIR_BIN" -p "$BACKUP_DIR"

# ── Resolve zip name collision ────────────────────────────────────────────────

counter=1
while [[ -e "$ZIP_PATH" ]]; do
  FINAL_NAME="${BASE_NAME}-${counter}"
  ZIP_NAME="${FINAL_NAME}.zip"
  ZIP_PATH="${BACKUP_DIR}/${ZIP_NAME}"
  counter=$((counter + 1))
done

# ── Staging area ──────────────────────────────────────────────────────────────

STAGING_DIR="$("$MKTEMP_BIN" -d)"
RENAMED_SOURCE="${STAGING_DIR}/${FINAL_NAME}"
FILE_LIST="${STAGING_DIR}/files.txt"
COPY_LIST="${STAGING_DIR}/copy_list.txt"

cleanup() {
  "$RM_BIN" -rf "$STAGING_DIR"
}
trap cleanup EXIT

# ── Progress bar ──────────────────────────────────────────────────────────────

print_progress() {
  local current=$1
  local total=$2
  local width=40
  local filled=0
  local percent=0
  local empty=0
  local bar=''

  if (( total > 0 )); then
    percent=$(( current * 100 / total ))
    filled=$(( current * width / total ))
  fi

  empty=$(( width - filled ))

  for ((i = 0; i < filled; i++)); do
    bar="${bar}#"
  done
  for ((i = 0; i < empty; i++)); do
    bar="${bar}-"
  done

  printf "\r[%s] %3d%% (%d/%d)" "$bar" "$percent" "$current" "$total"
}

# ── Build file list ───────────────────────────────────────────────────────────

"$MKDIR_BIN" -p "$RENAMED_SOURCE"
cd "$SOURCE_DIR"

echo "Building file list..."

if [[ "$IS_GIT_REPO" == true ]]; then
  "$GIT_BIN" ls-files --cached --others --exclude-standard | while IFS= read -r f; do
    # Skip files inside the backup dir itself
    [[ "$f" == backup/* ]] && continue
    [[ -e "$f" ]] && printf '%s\n' "$f"
  done > "$FILE_LIST"
else
  "$FIND_BIN" . \
    -path './backup' -prune -o \
    \( -name '.DS_Store' -o -name '._*' -o -name '.AppleDouble' -o -name '.LSOverride' \
       -o -name 'Icon?' -o -name '.Spotlight-V100' -o -name '.Trashes' \
       -o -name '.fseventsd' -o -name '.TemporaryItems' \
       -o -name '.ipynb_checkpoints' -o -name '__pycache__' \) -prune \
    -o -type f -print | sed 's|^\./||' > "$FILE_LIST"
fi

cp "$FILE_LIST" "$COPY_LIST"

# Add extra includes (gitignored directories you still want in the backup)
for path in "${EXTRA_INCLUDE[@]}"; do
  if [[ -e "$path" ]]; then
    if [[ -d "$path" ]]; then
      "$FIND_BIN" "$path" \
        \( -name '.DS_Store' -o -name '._*' -o -name '.AppleDouble' -o -name '.LSOverride' \
           -o -name 'Icon?' -o -name '.Spotlight-V100' -o -name '.Trashes' \
           -o -name '.fseventsd' -o -name '.TemporaryItems' \
           -o -name '.ipynb_checkpoints' -o -name '__pycache__' \) -prune \
        -o -type f -print >> "$COPY_LIST"
    else
      printf '%s\n' "$path" >> "$COPY_LIST"
    fi
  else
    echo "Warning: extra include path not found: $path"
  fi
done

"$AWK_BIN" '!seen[$0]++' "$COPY_LIST" > "${COPY_LIST}.tmp"
"$MV_BIN" "${COPY_LIST}.tmp" "$COPY_LIST"

TOTAL_FILES="$("$WC_BIN" -l < "$COPY_LIST" | "$TR_BIN" -d ' ')"

# ── Copy files ────────────────────────────────────────────────────────────────

echo "Copying $TOTAL_FILES files..."
current=0
while IFS= read -r f; do
  [[ -e "$f" ]] || continue
  target_dir="$RENAMED_SOURCE/$("$DIRNAME_BIN" "$f")"
  "$MKDIR_BIN" -p "$target_dir"
  "$CP_BIN" -pR "$f" "$target_dir/"
  current=$((current + 1))
  print_progress "$current" "$TOTAL_FILES"
done < "$COPY_LIST"
printf "\n"

# ── Strip macOS metadata ──────────────────────────────────────────────────────

echo "Cleaning metadata..."
"$FIND_BIN" "$RENAMED_SOURCE" \
  \( -name '.DS_Store' -o -name '._*' -o -name '.AppleDouble' -o -name '.LSOverride' \
     -o -name 'Icon?' -o -name '.Spotlight-V100' -o -name '.Trashes' \
     -o -name '.fseventsd' -o -name '.TemporaryItems' \
     -o -name '.ipynb_checkpoints' -o -name '__pycache__' \) \
  -exec "$RM_BIN" -rf {} + 2>/dev/null || true

if [[ -x "$XATTR_BIN" ]]; then
  "$XATTR_BIN" -rc "$RENAMED_SOURCE" 2>/dev/null || true
fi

# ── Create zip ────────────────────────────────────────────────────────────────

echo "Creating zip..."
cd "$STAGING_DIR"

[[ -e "$ZIP_PATH" ]] && "$RM_BIN" -f "$ZIP_PATH"

"$DITTO_BIN" -c -k --keepParent "$FINAL_NAME" "$ZIP_PATH"

echo "Created: $ZIP_PATH"
