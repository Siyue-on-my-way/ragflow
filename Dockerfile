FROM infiniflow/ragflow-base:v2.0
USER  root

WORKDIR /ragflow

ADD ./web ./web
RUN cd ./web && npm i --force && npm run build

# ADD ./api ./api
# ADD ./conf ./conf
# ADD ./deepdoc ./deepdoc
# ADD ./rag ./rag
# ADD ./agent ./agent
# ADD ./graphrag ./graphrag

ENV PYTHONPATH=/ragflow/
ENV HF_ENDPOINT=https://hf-mirror.com

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install jieba -i https://mirrors.aliyun.com/pypi/simple/
# RUN pip install --prefix=/usr/local -r /requirements.txt -i https://mirrors.aliyun.com/pypi/simple/


ADD docker/entrypoint.sh ./entrypoint.sh
ADD docker/.env ./
RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]