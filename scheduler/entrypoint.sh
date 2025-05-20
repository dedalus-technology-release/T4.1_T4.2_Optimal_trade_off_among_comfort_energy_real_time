#!/bin/sh

# Carica variabili da .env
if [ -f "/env/.env" ]; then
    export $(grep -v '^#' /env/.env | xargs)
fi

# Imposta default
: "${CRON_HOUR:=2}"
: "${CRON_MINUTE:=0}"

echo "⏰ Cron set at $CRON_HOUR:$CRON_MINUTE"

# Crea uno script che esegue le chiamate API
cat << 'EOF' > /usr/local/bin/run_tasks.sh
#!/bin/sh

# Carica le variabili da .env
if [ -f "/env/.env" ]; then
    export $(grep -v '^#' /env/.env | xargs)
fi

echo "🔐 Getting access token..."
RESPONSE=$(curl -s -X POST http://comfort-flexibility-service:8000/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=${ADMIN_USERNAME}&password=${ADMIN_PASSWORD}")

TOKEN=$(echo "$RESPONSE" | grep -o '"access_token":"[^"]*"' | cut -d':' -f2 | tr -d '"')

if [ -z "$TOKEN" ]; then
    echo "❌ Failed to get token"
    exit 1
fi

echo "✅ Token acquired"

echo "📡 Calling /forecast_sPMV..."
curl -s -H "Authorization: Bearer $TOKEN" http://comfort-flexibility-service:8000/forecast_sPMV

echo "📡 Calling /run_optimization..."
curl -s -H "Authorization: Bearer $TOKEN" http://comfort-flexibility-service:8000/run_optimization
EOF

chmod +x /usr/local/bin/run_tasks.sh

# Crea cronjob
echo "$CRON_MINUTE $CRON_HOUR * * * /usr/local/bin/run_tasks.sh >> /tmp/cron.log 2>&1" > /etc/crontabs/root

# Mostra crontab
cat /etc/crontabs/root

# Avvia dcron
crond -f -l 8
