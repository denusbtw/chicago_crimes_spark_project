FROM python:3.8-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y openjdk-17-jdk-headless && \
    apt-get clean;

RUN ln -s $(ls -d /usr/lib/jvm/java-17-openjdk-*) /usr/lib/jvm/java-default
ENV JAVA_HOME=/usr/lib/jvm/java-default
ENV PATH=$PATH:$JAVA_HOME/bin

RUN pip install --no-cache-dir pyspark

COPY . .

CMD ["python", "main.py"]