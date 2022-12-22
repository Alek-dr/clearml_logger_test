FROM python:3.9-slim

RUN apt-get update && apt-get upgrade -y
RUN pip install --upgrade pip

WORKDIR /home/app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD ["/home/app/pipeline_constraction.py"]
ENTRYPOINT ["python3"]