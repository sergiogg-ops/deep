FROM python:3.11.11

COPY translate.py translate.py
COPY requirements.txt requirements.txt
COPY train_pipe.py train_pipe.py

RUN pip install -r requirements.txt

CMD ["python","translate.py", "data/source.sgm", "data/predictions.sgm"]
