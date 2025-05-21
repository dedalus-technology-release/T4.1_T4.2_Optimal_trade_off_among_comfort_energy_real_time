#!/bin/sh

# Load variables from .env
if [ -f "/env/.env" ]; then
    export $(grep -v '^#' /env/.env | xargs)
fi

# Imposta default
: "${CRON_HOUR:=2}"
: "${CRON_MINUTE:=0}"

echo "⏰ Cron set at $CRON_HOUR:$CRON_MINUTE"

# Create script to run API calls.
cat << 'EOF' > /usr/local/bin/run_tasks.sh
#!/bin/sh

# Carica variabili
if [ -f "/env/.env" ]; then
    export $(grep -v '^#' /env/.env | xargs)
fi

COOKIE_FILE="/tmp/cookies.txt"

echo "🔐 Getting access token via cookie..."
curl -s -c "$COOKIE_FILE" -X POST http://comfort-flexibility-service:8000/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=${ADMIN_USERNAME}&password=${ADMIN_PASSWORD}"

echo "✅ Cookie content:"
cat "$COOKIE_FILE"

if ! grep -q "token" "$COOKIE_FILE"; then
    echo "❌ Failed to get valid token cookie. Exiting..."
    exit 1
fi

echo "📡 Calling /forecast_sPMV..."
curl -s -b "$COOKIE_FILE" http://comfort-flexibility-service:8000/forecast_sPMV

echo "📡 Calling /run_optimization..."
curl -s -b "$COOKIE_FILE" http://comfort-flexibility-service:8000/run_optimization

rm -f "$COOKIE_FILE"
EOF

chmod +x /usr/local/bin/run_tasks.sh

# create cronjob
echo "$CRON_MINUTE $CRON_HOUR * * * /usr/local/bin/run_tasks.sh >> /tmp/cron.log 2>&1" > /etc/crontabs/root

# show cron content for debugging
cat /etc/crontabs/root

# Run cron
crond -f -l 8
