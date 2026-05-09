"""Installers for macOS launchd, Linux systemd, and Windows Task
Scheduler.

Each installer only writes the unit/plist/XML file. The user runs the
actual load/enable command. We don't try to be smart about
authentication or service-account selection.
"""
from __future__ import annotations

import getpass
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from .config import expand, load_config


LAUNCHD_PLIST_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{binary}</string>
    <string>serve</string>
    <string>--port</string><string>{port}</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>{stdout}</string>
  <key>StandardErrorPath</key><string>{stderr}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
</dict>
</plist>
"""


def _resolve_binary(binary: str | None) -> str:
    if binary:
        return binary
    # Prefer the current interpreter's `llamanager` script if installed.
    candidate = shutil.which("llamanager")
    if candidate:
        return candidate
    # Fall back to `python -m llamanager` style.
    return f"{sys.executable}"


def install_launchd(*, label: str = "com.llamanager",
                    port: int | None = None,
                    binary: str | None = None) -> Path:
    cfg = load_config()
    binary_path = _resolve_binary(binary)
    p = port or cfg.port
    plist = LAUNCHD_PLIST_TEMPLATE.format(
        label=label,
        binary=binary_path,
        port=p,
        stdout=str(cfg.logs_dir / "llamanager.out"),
        stderr=str(cfg.logs_dir / "llamanager.err"),
    )
    target = expand(f"~/Library/LaunchAgents/{label}.plist")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(plist, encoding="utf-8")
    try:
        os.chmod(target, 0o644)
    except OSError:
        pass
    return target


SYSTEMD_UNIT_TEMPLATE = """\
[Unit]
Description=llamanager — manager and proxy for llama-server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart={binary} serve --port {port}
Restart=on-failure
RestartSec=5
StandardOutput=append:{stdout}
StandardError=append:{stderr}
Environment=PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin

[Install]
WantedBy=default.target
"""


def install_systemd(*, unit_name: str = "llamanager.service",
                    port: int | None = None,
                    binary: str | None = None) -> Path:
    cfg = load_config()
    binary_path = _resolve_binary(binary)
    p = port or cfg.port
    unit = SYSTEMD_UNIT_TEMPLATE.format(
        binary=binary_path,
        port=p,
        stdout=str(cfg.logs_dir / "llamanager.out"),
        stderr=str(cfg.logs_dir / "llamanager.err"),
    )
    target = expand(f"~/.config/systemd/user/{unit_name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(unit, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Windows Task Scheduler
# ---------------------------------------------------------------------------

WINDOWS_TASK_XML_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>llamanager — manager and proxy for llama-server</Description>
    <Author>{author}</Author>
    <Date>{now}</Date>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>{user}</UserId>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{user}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <DisallowStartOnRemoteAppSession>false</DisallowStartOnRemoteAppSession>
    <UseUnifiedSchedulingEngine>true</UseUnifiedSchedulingEngine>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>5</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{binary}</Command>
      <Arguments>{args}</Arguments>
      <WorkingDirectory>{cwd}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def install_windows_task(*, task_name: str = "llamanager",
                         port: int | None = None,
                         binary: str | None = None) -> Path:
    """Write a Windows Task Scheduler XML the user can register with
    `schtasks /create /xml <path> /tn llamanager`.

    Logon-triggered (runs at user logon, not system boot — matches the
    LaunchAgent semantics on macOS rather than a system service)."""
    cfg = load_config()
    binary_path = _resolve_binary(binary)
    p = port or cfg.port
    user = os.environ.get("USERDOMAIN", "") + "\\" + os.environ.get("USERNAME", getpass.getuser()) \
        if os.environ.get("USERDOMAIN") else os.environ.get("USERNAME", getpass.getuser())
    xml = WINDOWS_TASK_XML_TEMPLATE.format(
        author=xml_escape(user),
        now=datetime.now().isoformat(timespec="seconds"),
        user=xml_escape(user),
        binary=xml_escape(binary_path),
        args=xml_escape(f"serve --port {p}"),
        cwd=xml_escape(str(cfg.data_dir)),
    )
    target = cfg.data_dir / f"{task_name}.task.xml"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Task Scheduler expects UTF-16 LE with BOM (\xFF\xFE).
    target.write_bytes(b"\xff\xfe" + xml.encode("utf-16-le"))
    return target


__all__ = ["install_launchd", "install_systemd", "install_windows_task"]
