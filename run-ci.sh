TEST_DIR=tests
SOURCE_DIR=src/customer_churn

echo "---- SETTING UP POETRY ----"
pip install poetry
poetry install
poetry config virtualenvs.in-project false


echo "\n---- CODE STYLE CHECK ----"
# TODO: set up something to run black as matrix strategy
echo "Running Black over code base"
poetry run black --check --diff .

echo "\n---- LINTING ----"
echo "Running ruff over the source code"
poetry run ruff check .
if [[ -d "$TEST_DIR" ]]; then
    echo "Running ruff over the tests"
    poetry run ruff check $TEST_DIR
else
    echo "No tests found in directory '$TEST_DIR'"
fi
