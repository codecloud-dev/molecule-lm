#!/bin/bash
# molecule CI 监控（开发者本地用）：每分钟巡检 train/package/distill 活跃 run + release 列表 + VERSION。
# 仅在「状态变更 / 新发版 / failure」时写日志，避免刷屏。所有 gh 调用包 timeout 40 防卡死。
#
# 用法：
#   1) 先确保 GH_TOKEN 已设置（裸 token 即可走 API），例如写入 ~/.zshenv：
#        export GH_TOKEN="ghp_xxx"
#   2) 后台运行：  bash tools/ci_monitor.sh &    或    nohup bash tools/ci_monitor.sh >/tmp/ci_monitor.out 2>&1 &
#   3) 看日志：    tail -f /tmp/ci_monitor.log
#   4) 停掉：      找到本脚本 PID 后 kill（不要 pkill -f ci_monitor.sh，会误杀自身命令）
#
# 注意：本脚本只读不写，不会自动重触发任何 CI（避免失败死循环）。发现 failure 请人工排查。

set -u

LOG="${CI_MONITOR_LOG:-/tmp/ci_monitor.log}"
REPO="${CI_MONITOR_REPO:-codecloud-dev/molecule-lm}"

if [ -z "${GH_TOKEN:-}" ]; then
  echo "ERROR: 未设置 GH_TOKEN，请在 ~/.zshenv 或当前环境 export GH_TOKEN=..." >&2
  exit 1
fi
export GH_TOKEN

echo "$(date -u) MONITOR_START repo=$REPO" >> "$LOG"

prev=""
while true; do
  ts=$(date -u +%FT%TZ)
  snap=$(timeout 40 gh run list --repo "$REPO" --limit 14 \
    --json number,status,conclusion,workflowName,createdAt 2>/dev/null \
    | python3 -c "import sys,json
d=json.load(sys.stdin)
print('\n'.join(f\"#{r['number']} {r['status']:<11} {str(r['conclusion']):<10} {r['createdAt']} {r['workflowName']}\" for r in d))" 2>/dev/null)
  rel=$(timeout 40 gh release list --repo "$REPO" 2>/dev/null | head -12)
  ver=$(timeout 40 gh api "repos/$REPO/contents/VERSION" --jq '.content' 2>/dev/null \
        | base64 -d 2>/dev/null | tr -d '[:space:]')
  cur="$snap===REL===$rel===VER===$ver"
  if [ "$cur" != "$prev" ]; then
    {
      echo "$ts CHANGE"
      echo "$snap"
      [ -n "$rel" ] && echo "RELEASES: $rel"
      echo "VERSION(on main)=$ver"
      if echo "$snap" | grep -qE "failure|cancelled"; then
        echo "$ts ALERT: 有 run 失败/取消，需排查（不要自动重触发以免死循环）"
      fi
    } >> "$LOG"
    prev="$cur"
  fi
  sleep 60
done
