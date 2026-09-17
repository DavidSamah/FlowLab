#!/usr/bin/env bash
set -euo pipefail

printf '== FlowLab repository/CI trace ==\n'
printf 'branch: '; git branch --show-current
printf 'head:   '; git rev-parse HEAD
printf '\n-- changed from main --\n'
git diff --stat origin/main...HEAD || true

printf '\n-- required build inputs --\n'
required=(package.json tsconfig.json next-env.d.ts app/layout.tsx app/page.tsx app/globals.css postcss.config.mjs eslint.config.mjs .github/workflows/ci.yml)
for path in "${required[@]}"; do
  if [[ -e "$path" ]]; then printf 'OK      %s\n' "$path"; else printf 'MISSING %s\n' "$path"; fi
done

printf '\n-- reproducibility lock --\n'
if [[ -f package-lock.json ]]; then
  printf 'OK      package-lock.json\n'
else
  printf 'BOOTSTRAP package-lock.json absent; first CI pass must generate it before strict npm ci validation.\n'
fi

printf '\n-- workflow gates --\n'
sed -n '1,220p' .github/workflows/ci.yml 2>/dev/null || true

printf '\n-- package scripts --\n'
node -e 'const p=require("./package.json"); console.log(p.scripts)'
