#!/bin/bash
# Boot-time setup for the VoiceGym GPU worker: idle auto-stop.
# The box powers off (= EC2 "stopped", disk kept) after IDLE_MIN minutes with no
# GPU process, no SSH login, no tmux session and no /home/ubuntu/.keep-awake file.
cat > /usr/local/bin/voicegym-idle-check <<'SCRIPT'
#!/bin/bash
IDLE_MIN=30
STATE=/run/voicegym-idle-since
busy=0
[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && busy=1
[ -n "$(who)" ] && busy=1
sudo -u ubuntu tmux ls >/dev/null 2>&1 && busy=1
[ -e /home/ubuntu/.keep-awake ] && busy=1
if [ $busy = 1 ]; then rm -f $STATE; exit 0; fi
[ -f $STATE ] || date +%s > $STATE
if [ $(( $(date +%s) - $(cat $STATE) )) -ge $(( IDLE_MIN * 60 )) ]; then
  logger "voicegym: idle ${IDLE_MIN} min, powering off"
  shutdown -h now
fi
SCRIPT
chmod +x /usr/local/bin/voicegym-idle-check
cat > /etc/systemd/system/voicegym-idle.service <<'UNIT'
[Unit]
Description=VoiceGym idle auto-stop check
[Service]
Type=oneshot
ExecStart=/usr/local/bin/voicegym-idle-check
UNIT
cat > /etc/systemd/system/voicegym-idle.timer <<'UNIT'
[Unit]
Description=Run VoiceGym idle check every minute
[Timer]
OnBootSec=5min
OnUnitActiveSec=1min
[Install]
WantedBy=timers.target
UNIT
systemctl daemon-reload
systemctl enable --now voicegym-idle.timer
