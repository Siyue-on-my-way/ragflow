FROM infiniflow/ragflow-base:v2.0
USER  root

WORKDIR /ragflow


###########################################################
######  以下文件及文件夹，不在add, 而是从docker-compose中挂载
###########################################################
# ADD ./api ./api
# ADD ./conf ./conf
# ADD ./deepdoc ./deepdoc
# ADD ./rag ./rag
# ADD docker/entrypoint.sh ./entrypoint.sh
# ADD docker/.env ./
# ADD ./web ./web


#### 由此可见， build的时候挂载是不生效的，只能拉真实存在的数据， docker-composer的挂载仅对启动服务时生效
ADD ./web ./web
RUN cd /ragflow/web && npm i --force && npm run build

ENV PYTHONPATH=/ragflow/
ENV HF_ENDPOINT=https://hf-mirror.com

ADD docker/entrypoint.sh ./entrypoint.sh

RUN chmod 777 ./entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]