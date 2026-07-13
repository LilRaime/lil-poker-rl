import json
import socket
import requests
import websocket
from poker_env.base_env import BasePokerEnv
from poker_env.observation import encode_state
from adapters.lil_poker.auth import login_as_guest

_WS_RECV_TIMEOUT = 30


class LilPokerEnv(BasePokerEnv):
    def __init__(self, base_url: str, room_id: str, username: str):
        self.base_url = base_url.rstrip('/')
        self.room_id = room_id
        self.username = username

        print(f"Logging in to {self.base_url} as {username}...")
        self.session, self.user_info = login_as_guest(self.base_url, username)
        player_id = self.user_info["uuid"]
        print(f"Logged in successfully. Player ID: {player_id}")

        super().__init__(player_id=player_id, starting_chips=self.user_info.get("chips", 1000))

        self._join_room()
        self.ws = None
        self._connect_websocket()

    def _join_room(self):
        url = f"{self.base_url}/api/game/players?room={self.room_id}"
        response = self.session.post(url, json={"uuid": self.player_id})
        if response.status_code not in [200, 201]:
            print(f"Room join notice (status {response.status_code}): {response.text}")

    def _connect_websocket(self):
        ws_scheme = "ws" if self.base_url.startswith("http://") else "wss"
        domain = self.base_url.split("://")[-1]
        ws_url = f"{ws_scheme}://{domain}/api/game/ws?room={self.room_id}"

        cookies = self.session.cookies.get_dict()
        cookie_header = "; ".join([f"{k}={v}" for k, v in cookies.items()])
        headers = {"Cookie": cookie_header}

        print(f"Connecting to WS: {ws_url}")
        self.ws = websocket.create_connection(ws_url, header=headers)

    def _send_action(self, action: int, amount: int = 0) -> None:
        action_name, amount = self._map_action(action)
        url = f"{self.base_url}/api/game/act?room={self.room_id}"
        payload = {
            "player_id": self.player_id,
            "action": action_name,
            "amount": amount
        }
        response = self.session.post(url, json=payload)
        if response.status_code not in [200, 201]:
            raise RuntimeError(f"Failed to execute action {action_name}: {response.text}")

    def _drain_websocket(self) -> dict:
        self.ws.settimeout(0.01)
        last_state = None
        while True:
            try:
                msg = self.ws.recv()
                if msg:
                    state = json.loads(msg)
                    if "players" in state and "phase" in state:
                        last_state = state
            except (websocket.WebSocketTimeoutException, socket.timeout, BlockingIOError):
                break
        self.ws.settimeout(None)
        if last_state:
            self.game_state = last_state
        return last_state

    def _wait_for_my_turn(self, resetting: bool = False) -> dict:
        state = self._drain_websocket()
        if state:
            if state.get("phase") == "Waiting" and len(state.get("players", [])) >= 2:
                self._start_new_hand()
            if state.get("phase") not in ["Showdown", "Waiting"] and state.get("active_player_id") == self.player_id:
                return state
            if not resetting and state.get("phase") in ["Showdown", "Waiting"]:
                return state

        print("WS connected. Waiting for hand to start / my turn to act...", flush=True)
        self.ws.settimeout(_WS_RECV_TIMEOUT)
        while True:
            try:
                msg = self.ws.recv()
                if not msg:
                    raise ConnectionError("WS connection closed.")

                state = json.loads(msg)
                if "players" in state and "phase" in state:
                    self.game_state = state
                    if state.get("phase") == "Waiting" and len(state.get("players", [])) >= 2:
                        self._start_new_hand()
                    if state.get("phase") not in ["Showdown", "Waiting"] and state.get("active_player_id") == self.player_id:
                        self.ws.settimeout(None)
                        return state
                    if not resetting and state.get("phase") in ["Showdown", "Waiting"]:
                        self.ws.settimeout(None)
                        return state

            except (websocket.WebSocketTimeoutException, socket.timeout):
                print("WS idle timeout, sending keepalive ping...")
                try:
                    self.ws.ping()
                except Exception:
                    print("Ping failed. Reconnecting...")
                    self._connect_websocket()

            except Exception as e:
                print(f"Error reading from WS: {e}. Reconnecting...")
                try:
                    self._connect_websocket()
                except Exception as conn_err:
                    print(f"Reconnect failed: {conn_err}. Retrying in 5 s...")
                    import time
                    time.sleep(5)
                    self._connect_websocket()

    def reset(self, seed=None, options=None):
        my_status = self._get_player_status(self.game_state)
        self.last_chips = float(my_status.get("chips", self.starting_chips)) if my_status else self.starting_chips

        self.game_state = self._start_new_hand()
        self.game_state = self._wait_for_my_turn(resetting=True)

        obs = encode_state(self.game_state, self.player_id)
        info = {"game_state": self.game_state}
        return obs, info

    def _start_new_hand(self) -> dict:
        url = f"{self.base_url}/api/game/start?room={self.room_id}"
        try:
            response = self.session.post(url, timeout=10)
            if response.status_code not in [200, 201, 204, 403]:
                print(f"Warning: /start returned {response.status_code}: {response.text}")
        except requests.RequestException as e:
            print(f"Warning: failed to call /start: {e}")
        return self.game_state

    def _stand_up(self):
        url = f"{self.base_url}/api/game/stand?room={self.room_id}"
        try:
            self.session.post(url, timeout=5)
        except Exception as e:
            print(f"Error standing up: {e}")

    def close(self):
        self._stand_up()
        if self.ws:
            self.ws.close()
