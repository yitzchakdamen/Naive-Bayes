FROM python:3.11-slim
FROM model-lib__v2

WORKDIR /app

EXPOSE 8888

COPY . .

RUN pip3 install -r requirements.txt

CMD [ "python", "app.py" ]