#!/usr/bin/env bash
# Restart GAMA headless (port 1001) sạch trước mỗi lần train.
# Lý do: headless chạy tích lũy nhiều train-from-scratch sẽ XUỐNG CẤP → train diverge gridlock
# (signature: merger 100% timeout + mainline_mean_speed 0.000 + avg_reward 0.00). Restart = fix.
# Dùng: bash restart_headless.sh && ./.venv/Scripts/python.exe rl/train_marl.py ...
set -u
PORT="${1:-1001}"
GAMA_WD="C:/Users/dotie/AppData/Local/Programs/Gama/headless"

echo "[restart_headless] killing listener on port $PORT ..."
powershell -NoProfile -Command "\$c=Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue; if(\$c){Stop-Process -Id \$c.OwningProcess -Force; 'killed '+\$c.OwningProcess} else {'no listener'}"
sleep 2
echo "[restart_headless] relaunching gama-headless -socket $PORT ..."
powershell -NoProfile -Command "Start-Process -FilePath '$GAMA_WD/gama-headless.bat' -ArgumentList '-socket','$PORT' -WorkingDirectory '$GAMA_WD' -WindowStyle Hidden"

for i in $(seq 1 40); do
  if powershell -NoProfile -Command "try{(New-Object Net.Sockets.TcpClient).Connect('127.0.0.1',$PORT);exit 0}catch{exit 1}" 2>/dev/null; then
    echo "[restart_headless] READY on $PORT (waited ~$((i*8))s)"
    sleep 3   # đệm thêm cho GAMA sẵn sàng nhận experiment
    exit 0
  fi
  sleep 8
done
echo "[restart_headless] ERROR: headless NOT ready on $PORT after ~320s"
exit 1
