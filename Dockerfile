FROM python:3.11

RUN  pip install --upgrade pip

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD [ "python3", "/app/src/solution.py"]