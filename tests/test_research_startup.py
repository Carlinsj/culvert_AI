"""Research entry point only; no inference, training or listening sockets."""

import pytest

from scripts import serve_research
from scripts.serve_feature_staging import Handler


def test_research_reuses_the_existing_analysis_handler():
    assert serve_research.Handler is Handler


@pytest.mark.parametrize("port", ["-1", "0", "65536"])
def test_invalid_port_fails_before_binding(monkeypatch, port):
    def unexpected(*args):
        pytest.fail("Invalid ports must not start a server")

    monkeypatch.setattr(serve_research, "ThreadingHTTPServer", unexpected)
    with pytest.raises(SystemExit) as error:
        serve_research.main(["--port", port])
    assert error.value.code == 2


@pytest.mark.parametrize("arguments,port", [([], 8096), (["--port", "8098"], 8098)])
def test_loopback_research_startup_and_clean_shutdown(monkeypatch, capsys, arguments, port):
    events = []

    class Server:
        def __init__(self, address, handler):
            events.append((address, handler))

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            events.append("closed")

    monkeypatch.setattr(serve_research, "ThreadingHTTPServer", Server)
    serve_research.main(arguments)
    assert events == [(("127.0.0.1", port), Handler), "closed"]
    output = capsys.readouterr().out
    assert "Culvert AI research workspace" in output
    assert "summer 2027, subject to validation" in output
    assert "Recommendations require field verification." in output
    assert "staging" not in output.lower()


def test_occupied_port_gives_actionable_error_without_starting_other_apps(monkeypatch, capsys):
    def occupied(*args):
        raise OSError("Address already in use")

    monkeypatch.setattr(serve_research, "ThreadingHTTPServer", occupied)
    with pytest.raises(SystemExit) as error:
        serve_research.main([])
    assert error.value.code == 1
    message = capsys.readouterr().err
    assert "research workspace" in message and "--port" in message
    assert "Traceback" not in message
