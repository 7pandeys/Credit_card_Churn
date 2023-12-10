FROM python:3.10.13-slim

RUN pip install poetry

WORKDIR /vizion

COPY . .

RUN poetry install

RUN poetry config virtualenvs.create false

EXPOSE 80

# ENTRYPOINT ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8030" ]"
