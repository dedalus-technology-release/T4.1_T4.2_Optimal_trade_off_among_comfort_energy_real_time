FROM alpine:latest

# Installa solo dcron e curl
RUN apk update && apk add --no-cache curl dcron bash

# Copia lo script di entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
