#!/bin/bash
set -e

IMAGE_ID="366576748"
SSH_KEY_NAME="tenzo-key"
SIZE="cpx42"
REGION="nbg1"
SERVER_NAME="tenzo-build-node"
# UserKnownHostsFile=/dev/null — щоб нові сервери з тою ж IP не блокували known_hosts
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o LogLevel=ERROR"

# дістаємо hetzner api токен з локального конфігу
HCLOUD_TOKEN=$(grep '^\s*token\s*=' ~/.config/hcloud/cli.toml | head -1 | sed "s/.*= *'//;s/'.*//")

if [ -z "$HCLOUD_TOKEN" ]; then
    echo "-> ❌ HCLOUD_TOKEN порожній! Перевір ~/.config/hcloud/cli.toml"
    exit 1
fi

# Генерує watchdog-скрипт і завантажує на сервер, запускає через systemd
# Аргумент $1 — затримка перед першою перевіркою (секунди)
deploy_watchdog() {
    local INITIAL_DELAY="$1"
    local WATCHDOG_TMP
    WATCHDOG_TMP=$(mktemp /tmp/tenzo-watchdog.XXXXXX.sh)

    cat > "$WATCHDOG_TMP" <<WEOF
#!/bin/bash
# auto-generated watchdog for server $SERVER_ID

HCLOUD_TOKEN="$HCLOUD_TOKEN"
SERVER_ID="$SERVER_ID"

sleep $INITIAL_DELAY

while true; do
    ACTIVE=\$(ss -tnp | grep ':3632' | grep 'ESTAB' | wc -l)

    if [ "\$ACTIVE" -gt 0 ]; then
        echo "\$(date): distccd зайнятий (\$ACTIVE з'єднань), продовжуємо на 1 годину..."
        sleep 3600
    else
        echo "\$(date): distccd вільний, видаляємо сервер..."
        curl -s -X DELETE \
          -H "Authorization: Bearer \$HCLOUD_TOKEN" \
          "https://api.hetzner.cloud/v1/servers/\$SERVER_ID"
        break
    fi
done
WEOF

    scp $SSH_OPTS "$WATCHDOG_TMP" root@$SERVER_IP:/usr/local/bin/tenzo-watchdog.sh
    rm -f "$WATCHDOG_TMP"

    # systemd-run — єдиний 100% надійний спосіб запустити процес через SSH
    ssh $SSH_OPTS root@$SERVER_IP '
        chmod +x /usr/local/bin/tenzo-watchdog.sh
        systemctl stop tenzo-watchdog 2>/dev/null || true
        systemctl reset-failed tenzo-watchdog 2>/dev/null || true
        systemd-run --unit=tenzo-watchdog --remain-after-exit \
          /usr/local/bin/tenzo-watchdog.sh' || true
}

# 1. перевіряємо, чи сервер вже існує
if hcloud server describe $SERVER_NAME > /dev/null 2>&1; then
    echo "-> 🔄 знайдено активний сервер! використовуємо його..."
    SERVER_IP=$(hcloud server describe $SERVER_NAME -o format="{{.PublicNet.IPv4.IP}}")
    SERVER_ID=$(hcloud server describe $SERVER_NAME -o format="{{.ID}}")

    # Перевіряємо watchdog через systemd
    WATCHDOG_STATUS=$(ssh $SSH_OPTS root@$SERVER_IP '
        if systemctl is-active tenzo-watchdog >/dev/null 2>&1; then
            # Рахуємо скільки секунд юніт активний
            STARTED=$(systemctl show tenzo-watchdog -p ActiveEnterTimestampMonotonic --value 2>/dev/null)
            NOW=$(cat /proc/uptime | cut -d" " -f1 | cut -d. -f1)
            if [ -n "$STARTED" ] && [ "$STARTED" -gt 0 ] 2>/dev/null; then
                STARTED_SEC=$((STARTED / 1000000))
                ELAPSED=$((NOW - STARTED_SEC))
                echo "alive:${ELAPSED}"
            else
                echo "alive:0"
            fi
        else
            echo "dead:0"
        fi
    ' 2>/dev/null || echo "dead:0")

    WD_STATE=$(echo "$WATCHDOG_STATUS" | cut -d: -f1)
    WD_ELAPSED=$(echo "$WATCHDOG_STATUS" | cut -d: -f2)

    if [ "$WD_STATE" = "alive" ]; then
        WD_MINS=$((WD_ELAPSED / 60))
        echo "-> ✅ watchdog живий (працює ${WD_MINS} хв), збірка почнеться миттєво"
    else
        echo "-> ⚠️ watchdog мертвий! перезапускаємо (10 хв таймер)..."
        deploy_watchdog 600
        echo "-> ✅ watchdog перезапущено (видалить сервер через 10 хв простою)"
    fi
else
    echo "-> 🚀 піднімаємо новий сервер $SIZE (8 ядер) у hetzner..."
    hcloud server create --name $SERVER_NAME \
      --type $SIZE \
      --image $IMAGE_ID \
      --location $REGION \
      --ssh-key $SSH_KEY_NAME > /dev/null

    SERVER_IP=$(hcloud server describe $SERVER_NAME -o format="{{.PublicNet.IPv4.IP}}")
    SERVER_ID=$(hcloud server describe $SERVER_NAME -o format="{{.ID}}")
    echo "-> ✅ сервер готовий на ip: $SERVER_IP (id: $SERVER_ID)"

    echo "-> ⏳ чекаємо на запуск ssh..."
    for i in $(seq 1 30); do
        if ssh $SSH_OPTS root@$SERVER_IP true 2>/dev/null; then
            break
        fi
        sleep 1
    done

    # 2. розгортаємо watchdog з таймером 55 хвилин
    echo "-> ⏱️ розгортаємо auto-shutdown watchdog на сервері (55 хв таймер)..."
    deploy_watchdog 3300
    echo "-> ✅ watchdog запущено на сервері!"
fi

echo "-> 🔥 запускаємо віддалену компіляцію через docker..."
docker compose run --rm --remove-orphans \
  -e DISTCC_HOSTS="$SERVER_IP/32,lzo" \
  -e CCACHE_PREFIX="distcc" \
  dev ninja -C /app/cmake-build-debug tenzo-cli -j8

echo "-> 🎯 компіляцію завершено!"
echo "-> 💰 сервер залишається працювати. watchdog на сервері видалить його автоматично."