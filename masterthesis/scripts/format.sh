#!/bin/bash

# Append .gitignore to .prettierignore
\cp .prettierignore .prettierignore.bak
cat .gitignore >> .prettierignore

npx prettier --write .

# Restore .prettierignore
\cp .prettierignore.bak .prettierignore
rm .prettierignore.bak
