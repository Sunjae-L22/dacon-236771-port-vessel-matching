# author: opus
"""Message relay for the opus / argon / terra(Sol) discussion.

Every minute it looks for new message files in opus/, argon/ and terra/. A message wakes each agent named
on its To: line (never its own author):
  - terra (Sol): `codex exec` in the project folder (workspace-write sandbox)
  - argon: Antigravity CLI `agy -p` (terminal sandbox, shell only via allow rules); waits until agy is installed
  - opus: a line "OPUS_INBOX <file>" in relay.log; with shared/collab/OPUS_HEADLESS present also `claude -p`
    (at most OPUS_MAX_TURNS a day, since Claude runs on the smaller subscription)
Agents run one turn at a time each; several new messages for the same agent are handed over together.

Stop:   touch shared/collab/PAUSE   (remove it to resume)
Limits: MAX_TURNS per agent per day, COOLDOWN seconds between an agent's turns, TURN_TIMEOUT per turn.
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "shared" / "collab"
STATE = HERE / "state.json"
LOGS = HERE / "logs"
TIMELINE = HERE / "timeline.md"
FOLDERS = {"opus": "opus", "argon": "argon", "terra": "terra"}
NAMES = {"opus": "opus", "claude": "opus", "argon": "argon", "gemini": "argon", "terra": "terra", "sol": "terra"}
CODEX = "/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex"
AGY = str(Path.home() / ".local" / "bin" / "agy")
MAX_TURNS = int(os.environ.get("MAX_TURNS", "8"))
OPUS_MAX_TURNS = int(os.environ.get("OPUS_MAX_TURNS", "3"))  # Claude runs on the smaller subscription
OPUS_TOOLS = ("Read Glob Grep Write Edit Bash(python3:*) Bash(ls:*) Bash(cat:*) Bash(head:*) Bash(tail:*) "
              "Bash(grep:*) Bash(wc:*) Bash(unzip:*)")
COOLDOWN = int(os.environ.get("COOLDOWN", "300"))
TURN_TIMEOUT = int(os.environ.get("TURN_TIMEOUT", "2700"))
POLL = 60
WHO = {"terra": "Sol (ChatGPT). 당신의 폴더는 `terra/`입니다.", "argon": "Argon (Gemini). 당신의 폴더는 `argon/`입니다.",
       "opus": "Opus (Claude). 당신의 폴더는 `opus/`입니다."}


def log(msg):
    line = f"{datetime.now():%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)


def messages():
    """All message files as 'folder/name.md', oldest first by name within each folder."""
    out = []
    for f in FOLDERS.values():
        out += [f"{f}/{p.name}" for p in sorted((ROOT / f).glob("[0-9][0-9][0-9]_*.md"))]
    return out


def recipients(rel):
    author = rel.split("/")[0]
    head = (ROOT / rel).read_text(encoding="utf-8", errors="ignore").splitlines()[:12]
    to = next((l for l in head if re.match(r"\W*(To|대상)\W", l, re.I)), "")
    names = {NAMES[w] for w in re.findall(r"[a-z]+", to.lower()) if w in NAMES}
    return sorted(n for n in names if n != author)


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"seen": messages(), "pending": {}, "turns": {}}  # start from now, not the backlog


def save_state(st):
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1))


def prompt(agent, files):
    rules = (HERE / "AGENT_RULES.md").read_text(encoding="utf-8")
    listing = "\n".join(f"- `{f}`" for f in files)
    return (f"{rules}\n\n## 이번 차례\n\n당신은 {WHO[agent]} 작업 폴더(루트)는 `{ROOT}`입니다.\n\n"
            f"새 메시지:\n{listing}\n")


def clean_env():
    return {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "ANTHROPIC"))}


_login = {"t": 0.0, "ok": False}


def claude_logged_in():
    """`claude auth status` on a clean environment, re-checked at most every 5 minutes."""
    if time.time() - _login["t"] > 300:
        r = subprocess.run(["claude", "auth", "status"], capture_output=True, text=True, env=clean_env())
        try:
            _login["ok"] = bool(json.loads(r.stdout).get("loggedIn"))
        except ValueError:
            _login["ok"] = False
        _login["t"] = time.time()
        if not _login["ok"]:
            log("opus waiting: claude CLI is not logged in")
    return _login["ok"]


def start(agent, files):
    ts = datetime.now().strftime("%m%d-%H%M%S")
    out = open(LOGS / f"{agent}_{ts}.log", "w")
    p = prompt(agent, files)
    if agent == "terra":
        cmd = [CODEX, "exec", "-C", str(ROOT), "--skip-git-repo-check", "-s", "workspace-write",
               "-o", str(LOGS / f"terra_{ts}.last.txt"), p]
        env = dict(os.environ)
    else:  # Antigravity CLI (Gemini CLI sign-in no longer works for personal accounts)
        # No blanket auto-approve: workspace file reads/writes are allowed by default, shell commands only
        # through permissions.allow rules in ~/.gemini/antigravity-cli/settings.json (others are soft-denied).
        cmd = [AGY, "-p", p, "--sandbox", "--print-timeout", f"{TURN_TIMEOUT}s"]
        if os.environ.get("ARGON_MODEL"):
            cmd += ["--model", os.environ["ARGON_MODEL"]]
        env = dict(os.environ)
    if agent == "opus":  # Claude Code CLI on the user's own login; drop variables a parent Claude session leaves
        cmd = ["claude", "-p", p, "--permission-mode", "acceptEdits", "--allowedTools", OPUS_TOOLS]
        env = clean_env()
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            env=env), time.time()


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    st = load_state()
    running = {}
    last_end = {}
    log(f"relay started; root={ROOT}; seen={len(st['seen'])}")
    while True:
        today = datetime.now().strftime("%Y-%m-%d")
        for rel in messages():
            if rel in st["seen"]:
                continue
            st["seen"].append(rel)
            to = recipients(rel)
            with open(TIMELINE, "a", encoding="utf-8") as t:
                t.write(f"- {datetime.now():%m-%d %H:%M} `{rel}` → {', '.join(to) or '(수신자 없음)'}\n")
            for r in to:
                log(f"OPUS_INBOX {rel}" if r == "opus" else f"queued {rel} for {r}")
                if r != "opus" or (HERE / "OPUS_HEADLESS").exists():
                    st["pending"].setdefault(r, []).append(rel)
        for agent, (proc, t0) in list(running.items()):
            if proc.poll() is not None:
                log(f"{agent} turn finished rc={proc.returncode} in {time.time() - t0:.0f}s")
                del running[agent]
                last_end[agent] = time.time()
            elif time.time() - t0 > TURN_TIMEOUT:
                proc.kill()
                log(f"{agent} turn killed after {TURN_TIMEOUT}s")
                del running[agent]
                last_end[agent] = time.time()
        paused = (HERE / "PAUSE").exists()
        for agent, files in list(st["pending"].items()):
            if not files or agent in running or paused:
                continue
            if agent == "argon" and not (HERE / "ARGON_READY").exists():
                continue  # Argon stays manual until agy is signed in and this marker is created
            if agent == "opus" and not claude_logged_in():
                continue  # keep Opus's messages queued instead of spending its few turns on auth errors
            if time.time() - last_end.get(agent, 0) < COOLDOWN:
                continue
            key = f"{agent}:{today}"
            cap = OPUS_MAX_TURNS if agent == "opus" else MAX_TURNS
            if st["turns"].get(key, 0) >= cap:
                continue
            st["turns"][key] = st["turns"].get(key, 0) + 1
            running[agent] = start(agent, files)
            log(f"{agent} turn {st['turns'][key]}/{cap} started for {files}")
            st["pending"][agent] = []
        save_state(st)
        time.sleep(POLL)


if __name__ == "__main__":
    sys.exit(main())
