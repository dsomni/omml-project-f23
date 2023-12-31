#!/bin/bash
echo "#!/bin/sh
pipenv run pip freeze > requirements.txt
pipenv run ruff format .
pipenv run ruff check . --fix --exit-non-zero-on-fix
FILES=\$(git diff --diff-filter=d --name-only)
git add \$FILES requirements.txt" > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit