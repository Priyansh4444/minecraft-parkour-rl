import io
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from parkour.minecraft.server import ServerOptions, Spectator, player_name, run_server, validate


def joined(name):
    return f"[12:00:00] [Server thread/INFO]: {name} joined the game\n"


def left(name):
    return f"[12:00:00] [Server thread/INFO]: {name} left the game\n"


class SpectatorTests(unittest.TestCase):
    def test_names_and_viewer_must_differ_from_bot(self):
        for name in ("", "x\nop Admin", "with space", "a" * 17, "@a", "Name;stop"):
            with self.assertRaises(ValueError):
                player_name(name)
        self.assertEqual(player_name("My_Player1"), "My_Player1")
        with self.assertRaises(ValueError):
            Spectator("watcher", "Watcher")

    def test_both_join_orders_and_no_commands_control_the_bot(self):
        expected = ("gamemode spectator Viewer", "tp Viewer Watcher", "spectate Watcher Viewer")
        for first, second in (("Viewer", "Watcher"), ("Watcher", "Viewer")):
            watcher = Spectator("Viewer", "Watcher")
            self.assertEqual(watcher.on_log(joined(first)), ())
            self.assertEqual(watcher.on_log(joined(second)), expected)
            self.assertEqual(watcher.on_log(joined(first)), ())
            self.assertEqual(watcher.on_log(joined("Other")), ())

    def test_reconnect_of_either_player_reattaches(self):
        for name in ("Viewer", "Watcher"):
            watcher = Spectator("Viewer", "Watcher")
            watcher.on_log(joined("Viewer"))
            original = watcher.on_log(joined("Watcher"))
            self.assertEqual(watcher.on_log(left(name)), ())
            self.assertEqual(watcher.on_log(joined(name)), original)

    def test_player_chat_cannot_spoof_a_join(self):
        watcher = Spectator("Viewer", "Watcher")
        self.assertEqual(watcher.on_log("[Server thread/INFO]: <Other> Viewer joined the game"), ())
        self.assertEqual(watcher.online, set())


class ServerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        (self.directory / "server.jar").write_bytes(b"fixture")
        (self.directory / "server.properties").write_text("server-ip=127.0.0.1\n")

    def test_missing_jar_and_nonlocal_bind_are_rejected(self):
        options = ServerOptions(self.directory)
        (self.directory / "server.properties").write_text("server-ip=\n")
        with self.assertRaises(ValueError):
            validate(options)
        (self.directory / "server.jar").unlink()
        with self.assertRaises(FileNotFoundError):
            validate(options)

    def test_occupied_port_leaves_existing_listener_alone(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with self.assertRaises(RuntimeError):
                validate(ServerOptions(self.directory, port=listener.getsockname()[1]))
            self.assertGreater(listener.fileno(), 0)

    def process(self, lines):
        process = Mock()
        process.stdout = io.StringIO("".join(lines))
        process.stdin = Mock()
        process.poll.return_value = None
        process.wait.return_value = 0
        return process

    def test_headless_server_only_grants_bot_operator(self):
        process = self.process(
            ["[Server thread/INFO]: Done (1.0s)!\n", joined("Viewer"), joined("Watcher")]
        )
        with (
            patch("parkour.minecraft.server.validate"),
            patch("parkour.minecraft.server.subprocess.Popen", return_value=process),
            patch("sys.stdout", new=io.StringIO()),
        ):
            self.assertEqual(run_server(ServerOptions(self.directory)), 0)
        self.assertEqual(
            [call.args[0] for call in process.stdin.write.call_args_list],
            ["op Watcher\n", "stop\n"],
        )

    def test_visible_server_attaches_only_viewer_and_stops_owned_process(self):
        process = self.process(
            ["[Server thread/INFO]: Done (1.0s)!\n", joined("Viewer"), joined("Watcher")]
        )
        with (
            patch("parkour.minecraft.server.validate"),
            patch("parkour.minecraft.server.subprocess.Popen", return_value=process) as spawn,
            patch("sys.stdout", new=io.StringIO()),
        ):
            self.assertEqual(run_server(ServerOptions(self.directory, watch_player="Viewer")), 0)
        commands = [call.args[0] for call in process.stdin.write.call_args_list]
        self.assertEqual(
            commands,
            [
                "op Watcher\n",
                "gamemode spectator Viewer\n",
                "tp Viewer Watcher\n",
                "spectate Watcher Viewer\n",
                "stop\n",
            ],
        )
        self.assertEqual(spawn.call_args.args[0][-3:], ["nogui", "--port", "25566"])
        self.assertTrue(spawn.call_args.kwargs["start_new_session"])
        process.wait.assert_any_call(timeout=30)

    def test_interrupt_saves_and_stops_only_owned_process(self):
        process = self.process([])
        process.stdout = Mock()
        process.stdout.__iter__ = Mock(side_effect=KeyboardInterrupt)
        with (
            patch("parkour.minecraft.server.validate"),
            patch("parkour.minecraft.server.subprocess.Popen", return_value=process),
            patch("sys.stdout", new=io.StringIO()),
        ):
            self.assertEqual(run_server(ServerOptions(self.directory)), 0)
        process.stdin.write.assert_called_once_with("stop\n")
        process.wait.assert_called_once_with(timeout=30)


if __name__ == "__main__":
    unittest.main()
