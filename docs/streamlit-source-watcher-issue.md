# Draft bug report for streamlit/streamlit

Not filed. Sections follow the repository's bug report form (Checklist, Summary, Reproducible Code Example, Steps To Reproduce, Expected Behavior, Current Behavior, Is this a regression?, Debug info, Additional Information). Only Checklist and Summary are required.

Background for this repo: the app works around it in `artalk_streamlit_realtime/streamlit_patches.py`, and the stalls were originally found with the watchdog in `artalk_streamlit_realtime/loop_watchdog.py`. The report below is written to stand alone, so it uses a self-contained reproduction rather than anything from this project.

## Title

`update_watched_modules()` stats every `sys.modules` path on the event loop, blocking the server for seconds

## Checklist

All three items apply: existing issues searched, descriptive title, sufficient information to reproduce.

## Summary

`LocalSourcesWatcher.update_watched_modules()` is called inline on the event loop from `AppSession._handle_scriptrunner_event_on_event_loop` (`app_session.py:714`). It calls `get_module_paths()` for every entry in `sys.modules`, and every path returned goes through `os.path.realpath()` plus `_is_valid_path()`, which does `os.path.isfile()` or `os.path.isdir()`. The event loop therefore performs filesystem work proportional to the number of loaded module paths, and it is the same loop that serves every HTTP request and every WebSocket connection.

The rescan fires whenever `set(sys.modules)` changes, so it is not limited to startup: any lazy import during a session triggers a full rescan of every module. While it runs, the server answers nothing. In the reproduction below, a request to Streamlit's own `/_stcore/health` endpoint took 3432 ms against a median of 1.2 ms, and disabling only this scan brings the maximum down to 60 ms.

## Reproducible Code Example

`app.py`, which needs no third party dependencies. It creates a package of small modules to stand in for a large dependency stack, then imports one more module every few seconds so that `set(sys.modules)` keeps changing:

```python
import os
import pathlib
import sys
import time

import streamlit as st

if os.environ.get("SKIP_MODULE_SCAN"):
    # Control: the same app with the scan neutralised.
    from streamlit.watcher.local_sources_watcher import LocalSourcesWatcher

    LocalSourcesWatcher.update_watched_modules = lambda self: None

MODULES = 3000
PKG = pathlib.Path(__file__).parent / "fake_deps"


@st.cache_resource
def build_and_import() -> int:
    PKG.mkdir(exist_ok=True)
    (PKG / "__init__.py").write_text("")
    for i in range(MODULES):
        (PKG / f"mod{i}.py").write_text(f"value = {i}\n")
    sys.path.insert(0, str(PKG.parent))
    for i in range(MODULES):
        __import__(f"{PKG.name}.mod{i}")
    return len(sys.modules)


st.write(f"{build_and_import()} modules loaded at startup")
st.write(f"{len(sys.modules)} modules now")


@st.fragment(run_every="3s")
def import_one_more() -> None:
    i = st.session_state.get("n", 0)
    st.session_state["n"] = i + 1
    (PKG / f"extra{i}.py").write_text("")
    __import__(f"{PKG.name}.extra{i}")
    st.write(f"imported extra{i} at {time.strftime('%H:%M:%S')}")


import_one_more()
```

## Steps To Reproduce

1. `streamlit run app.py`
2. Open the app in a browser, so that a session exists and the script runs.
3. In another terminal, measure how long the server takes to answer a trivial request that is served by the event loop:

   ```sh
   while true; do
     curl -s -o /dev/null -w "%{time_total}\n" http://localhost:8501/_stcore/health
     sleep 0.1
   done
   ```

4. Watch the reported times. Most responses are around a millisecond, and the ones landing during a rescan take hundreds of milliseconds to seconds.
5. For the control, stop the server and start it again with `SKIP_MODULE_SCAN=1 streamlit run app.py`. Everything else is identical, only `update_watched_modules()` is neutralised.

## Expected Behavior

Resolving module paths does not block the event loop, so the server keeps answering requests while a rescan happens. Setting `server.fileWatcherType = "none"` skips the work entirely.

## Current Behavior

The loop is blocked for the duration of the scan. Latency of `/_stcore/health`, sampled about ten times a second over 75 s runs of the app above:

| run | samples | median | p99 | max | responses over 200 ms |
| --- | --- | --- | --- | --- | --- |
| default | 732 | 1.2 ms | 7 ms | 3432 ms | 1 |
| `SKIP_MODULE_SCAN=1` | 756 | 1.2 ms | 4 ms | 60 ms | 0 |

The two runs differ only in whether `update_watched_modules()` does its work, so the stall is attributable to the scan rather than to the imports themselves.

Stack captured with `faulthandler` from a separate thread while the loop was still blocked, on the thread running `bootstrap.run`:

```
File "streamlit/runtime/app_session.py", line 587 in <lambda>
File "streamlit/runtime/app_session.py", line 714 in _handle_scriptrunner_event_on_event_loop
File "streamlit/watcher/local_sources_watcher.py", line 272 in update_watched_modules
File "streamlit/watcher/local_sources_watcher.py", line 340 in get_module_paths
File "streamlit/watcher/local_sources_watcher.py", line 346 in _is_valid_path
File "<frozen genericpath>", line 30 in isfile
```

The absolute cost depends on how many modules are loaded and on how fast the filesystem answers `stat`. Scanning the paths of a `torch` plus `transformers` import (1682 paths) was measured between 340 ms and 13666 ms on one machine with the environment on an NFS mount, the spread tracking attribute cache state rather than anything about the app: the same process scanning twice took 350 ms immediately after importing and 1662 ms after sitting idle for 120 s. Network filesystems make it worse, but even locally the work is unbounded and on the loop.

## Is this a regression?

Not verified against older versions. The code path looks long standing rather than newly introduced.

## Debug info

Streamlit 1.58.0, Python 3.12.2, Linux 5.15.0 (glibc 2.35).

## Additional Information

Setting `server.fileWatcherType = "none"` does not avoid the cost. The `NoOpPathWatcher` short circuit lives in `_register_watcher` (`local_sources_watcher.py:222`), which runs after `update_watched_modules()` has already resolved every module path. Users who disable file watching for performance still pay for the scan in full.

Two possible fixes, either of which would remove the stall without changing behavior:

1. Return early from `update_watched_modules()` when the resolved path watcher class is `NoOpPathWatcher`, so the config setting does what it appears to promise.
2. Run the scan in an executor rather than on the event loop, so the loop stays responsive regardless of module count and filesystem latency.
