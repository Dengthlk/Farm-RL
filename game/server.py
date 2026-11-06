"""Local HTML farm game. Start this file with the bundled Python runtime."""
import argparse
import copy
import json
import os
import threading
import uuid
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
from urllib.request import urlopen

from engine.match import Match
from opponents import OPPONENTS, model_agent

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OMP_NUM_THREADS", "1")


class Game:
    def __init__(self, seed, opponent):
        self.id = uuid.uuid4().hex
        self.seed = seed
        self.opponent = opponent
        self.match = Match(seed)
        self.agent = self.match.agents["starter"] if opponent == "starter" else model_agent(opponent)
        self.history = []
        self.lock = threading.Lock()

    def snapshot(self):
        match = self.match
        return dict(id=self.id, seed=self.seed, opponent=self.opponent,
                    observation=match.state[0].observation, done=match.done,
                    rewards=[seat.reward for seat in match.state] if match.done else None,
                    animations=match.animations, history=self.history[-40:],
                    totalTurns=match.configuration.episodeSteps - 1)

    def play(self, step, action):
        match = self.match
        before = copy.deepcopy(match.state[0].observation)
        if match.done or step != before.step:
            raise ValueError("回合已更新，请刷新页面继续。")
        reply = self.agent(copy.deepcopy(match.state[1].observation))
        match.step([action, reply])
        after = match.state[0].observation
        self.history.append(dict(step=step, day=before.day, hour=before.hour, action=action,
                                 delta=after.farms[0]["money"] - before.farms[0]["money"],
                                 changed=before.farms[0] != after.farms[0] or before.private != after.private))
        return self.snapshot()


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "site"), **kwargs)

    def send_json(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == "/api/health":
            return self.send_json(dict(game="Farm-RL-Game", opponents=OPPONENTS))
        if url.path in ("/api/game", "/api/export"):
            game = self.server.games.get(parse_qs(url.query).get("id", [""])[0])
            if not game:
                return self.send_json({"error": "请开启一个新赛季。"}, 404)
            with game.lock:
                data = game.snapshot()
                if url.path == "/api/export":
                    data["humanActions"] = [row["action"] for row in game.history]
                return self.send_json(data)
        super().do_GET()

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/api/new":
                opponent = body.get("opponent", "starter")
                if opponent not in OPPONENTS:
                    raise ValueError("请选择一个农场对手。")
                game = Game(int(body.get("seed", 42)), opponent)
                self.server.games[game.id] = game
                return self.send_json(game.snapshot())
            if self.path == "/api/step":
                game = self.server.games[body["id"]]
                with game.lock:
                    return self.send_json(game.play(body["step"], body["action"]))
            self.send_json({"error": "没有这个操作。"}, 404)
        except (ValueError, KeyError, TypeError) as error:
            self.send_json({"error": str(error)}, 400)
        except Exception as error:
            self.send_json({"error": f"对局出错：{error}"}, 500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--no-browser", action="store_true")
    options = parser.parse_args()
    def open_game(port):
        url = f"http://127.0.0.1:{port}/"
        print("Farm RL is ready:", url, flush=True)
        if not options.no_browser:
            webbrowser.open(url)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", options.port), Handler)
    except OSError:
        try:
            with urlopen(f"http://127.0.0.1:{options.port}/api/health", timeout=1) as response:
                existing = json.load(response).get("game")
        except (OSError, ValueError):
            existing = None
        if existing == "Farm-RL-Game":
            open_game(options.port)
            return
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.games = {}
    open_game(server.server_port)
    server.serve_forever()


if __name__ == "__main__":
    main()

