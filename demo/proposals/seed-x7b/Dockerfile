FROM python:3.11.11

COPY translate.py translate.py
COPY requirements.txt requirements.txt
COPY translate.sh translate.sh

RUN chmod +x translate.sh
RUN pip install -r requirements.txt

CMD ["bash","translate.sh", "data/source.sgm", "data/predictions.sgm"]