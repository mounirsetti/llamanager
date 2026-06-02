"""Installers for macOS launchd, Linux systemd, and Windows Task
Scheduler.

Each installer only writes the unit/plist/XML file. The user runs the
actual load/enable command. We don't try to be smart about
authentication or service-account selection.

Every writer accepts an optional ``cfg`` so a caller that already loaded
a (possibly non-default) config can pass it through, instead of each
writer silently re-reading the default ``~/.llamanager/config.toml``.
"""
from __future__ import annotations

import getpass
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from .config import Config, expand, load_config


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
{program_args}
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


def _plist_program_args(command: str, port: int) -> str:
    """Render the `serve --port N` (daemon) or `tray` (thin client) argv
    as indented <string> elements for the plist ProgramArguments array."""
    if command == "tray":
        argv = ["tray"]
    else:
        argv = ["serve", "--port", str(port)]
    return "\n".join(f"    <string>{xml_escape(a)}</string>" for a in argv)


def install_launchd(*, label: str = "com.llamanager",
                    port: int | None = None,
                    binary: str | None = None,
                    command: str = "serve",
                    cfg: Config | None = None) -> Path:
    """Write a per-user LaunchAgent.

    ``command="serve"`` (default) installs the always-on daemon;
    ``command="tray"`` installs the thin-client tray that talks to it.
    Use a distinct ``label`` for the tray agent so both can coexist
    (e.g. ``com.llamanager.tray``)."""
    cfg = cfg or load_config()
    binary_path = _resolve_binary(binary)
    p = port or cfg.port
    plist = LAUNCHD_PLIST_TEMPLATE.format(
        label=label,
        binary=binary_path,
        program_args=_plist_program_args(command, p),
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


# ---------------------------------------------------------------------------
# macOS system LaunchDaemon — the only way to run the daemon *before login*
# on macOS (a per-user LaunchAgent starts at login). Lives in
# /Library/LaunchDaemons (root-owned), so installing it needs sudo. We run
# the process as the invoking user (UserName) rather than root, which is
# safer and keeps the data dir / GPU access under the right account.
#
# Trade-off vs. the LaunchAgent: controlling a system daemon (start/stop)
# also needs root, so the tray's daemon controls will require sudo. service_ctl
# detects the system daemon and reports this rather than failing silently.
# ---------------------------------------------------------------------------

LAUNCHDAEMON_DIR = Path("/Library/LaunchDaemons")

LAUNCHDAEMON_PLIST_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{label}</string>
  <key>UserName</key><string>{username}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{binary}</string>
{program_args}
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


def _invoking_user() -> str:
    """The human user, even when we're running under sudo."""
    return os.environ.get("SUDO_USER") or os.environ.get("USER") or getpass.getuser()


def install_launchdaemon(*, label: str = "com.llamanager",
                         port: int | None = None,
                         binary: str | None = None,
                         username: str | None = None,
                         cfg: Config | None = None) -> tuple[Path, Path]:
    """Write a system LaunchDaemon plist to a *staging* file (writable
    without root) and return ``(staging_path, system_dest)``.

    The caller installs it with ``sudo cp <staging> <dest>`` +
    ``sudo launchctl bootstrap system <dest>``. Runs the daemon at boot,
    before login, as ``username`` (default: the invoking user)."""
    cfg = cfg or load_config()
    binary_path = _resolve_binary(binary)
    p = port or cfg.port
    user = username or _invoking_user()
    plist = LAUNCHDAEMON_PLIST_TEMPLATE.format(
        label=label,
        username=xml_escape(user),
        binary=binary_path,
        program_args=_plist_program_args(command="serve", port=p),
        stdout=str(cfg.logs_dir / "llamanager.out"),
        stderr=str(cfg.logs_dir / "llamanager.err"),
    )
    staging = cfg.data_dir / f"{label}.daemon.plist"
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.write_text(plist, encoding="utf-8")
    return staging, LAUNCHDAEMON_DIR / f"{label}.plist"


SYSTEMD_UNIT_TEMPLATE = """\
[Unit]
Description=llamanager — manager and proxy for llama-server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart={binary} {exec_args}
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
                    binary: str | None = None,
                    command: str = "serve",
                    cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    binary_path = _resolve_binary(binary)
    p = port or cfg.port
    exec_args = "tray" if command == "tray" else f"serve --port {p}"
    unit = SYSTEMD_UNIT_TEMPLATE.format(
        binary=binary_path,
        exec_args=exec_args,
        stdout=str(cfg.logs_dir / "llamanager.out"),
        stderr=str(cfg.logs_dir / "llamanager.err"),
    )
    target = expand(f"~/.config/systemd/user/{unit_name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(unit, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Linux desktop autostart (XDG) — the right home for the *tray* on Linux,
# since the tray needs a graphical session. The daemon stays a
# systemd --user unit (install_systemd); this just launches the tray at
# login via a .desktop entry in ~/.config/autostart.
# ---------------------------------------------------------------------------

XDG_AUTOSTART_TEMPLATE = """\
[Desktop Entry]
Type=Application
Name=llamanager tray
Comment=System-tray controller for the llamanager daemon
Exec={binary} tray
Terminal=false
X-GNOME-Autostart-enabled=true
"""


def install_xdg_autostart(*, binary: str | None = None) -> Path:
    """Write ~/.config/autostart/llamanager-tray.desktop so the tray
    launches at desktop login."""
    binary_path = _resolve_binary(binary)
    entry = XDG_AUTOSTART_TEMPLATE.format(binary=binary_path)
    target = expand("~/.config/autostart/llamanager-tray.desktop")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(entry, encoding="utf-8")
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
                         binary: str | None = None,
                         command: str = "serve",
                         cfg: Config | None = None) -> Path:
    """Write a Windows Task Scheduler XML the user can register with
    `schtasks /create /xml <path> /tn llamanager`.

    Logon-triggered (runs at user logon, not system boot — matches the
    LaunchAgent semantics on macOS rather than a system service).

    ``command="tray"`` writes a task that launches the thin-client tray
    instead of the daemon — use this alongside the boot-time Windows
    service (install-windows-service) for the always-on + icon setup."""
    cfg = cfg or load_config()
    binary_path = _resolve_binary(binary)
    p = port or cfg.port
    task_args = "tray" if command == "tray" else f"serve --port {p}"
    user = os.environ.get("USERDOMAIN", "") + "\\" + os.environ.get("USERNAME", getpass.getuser()) \
        if os.environ.get("USERDOMAIN") else os.environ.get("USERNAME", getpass.getuser())
    xml = WINDOWS_TASK_XML_TEMPLATE.format(
        author=xml_escape(user),
        now=datetime.now().isoformat(timespec="seconds"),
        user=xml_escape(user),
        binary=xml_escape(binary_path),
        args=xml_escape(task_args),
        cwd=xml_escape(str(cfg.data_dir)),
    )
    target = cfg.data_dir / f"{task_name}.task.xml"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Task Scheduler expects UTF-16 LE with BOM (\xFF\xFE).
    target.write_bytes(b"\xff\xfe" + xml.encode("utf-16-le"))
    return target


# ---------------------------------------------------------------------------
# Windows: grant the current (non-admin) account the right to start/stop/
# query the daemon service, so the tray can control it WITHOUT a UAC prompt.
# Run this once, from the elevated install step.
# ---------------------------------------------------------------------------


def grant_windows_service_control(service_name: str = "llamanager",
                                  account: str | None = None) -> tuple[bool, str]:
    """Append an ACE to the service's security descriptor granting the
    account RP (start), WP (stop), DT (pause/continue), LC/RC (query/read)
    on the service. Must run elevated (it modifies the SD).

    Returns ``(ok, message)``. No-op-safe to re-run."""
    # Resolve the account SID. Default: current interactive user.
    acct = account or (
        (os.environ.get("USERDOMAIN", "") + "\\" if os.environ.get("USERDOMAIN")
         else "") + os.environ.get("USERNAME", getpass.getuser())
    )
    # Read the current SDDL.
    show = _run_sc(["sdshow", service_name])
    if show.returncode != 0:
        return (False, f"sc sdshow failed: {show.stdout or show.stderr}".strip())
    current = (show.stdout or "").strip()
    if not current.startswith("D:"):
        return (False, f"unexpected SDDL from sdshow: {current!r}")

    sid = _lookup_sid(acct)
    if sid is None:
        return (False, f"could not resolve SID for account {acct!r}")
    ace = f"(A;;RPWPDTLCRC;;;{sid})"
    if ace in current:
        return (True, f"control already granted to {acct}")

    # Insert the ACE into the DACL (after the leading "D:" + flags, before
    # the first "(" of the existing ACE list, or at the SACL boundary "S:").
    dacl, _, sacl = current.partition("S:")
    new_dacl = dacl + ace
    new_sddl = new_dacl + ("S:" + sacl if sacl else "")
    setres = _run_sc(["sdset", service_name, new_sddl])
    if setres.returncode != 0:
        return (False, f"sc sdset failed: {setres.stdout or setres.stderr}".strip())
    return (True, f"granted start/stop/query on {service_name} to {acct}")


def _run_sc(args: list[str]):
    import subprocess
    return subprocess.run(["sc", *args], capture_output=True, text=True,
                          check=False, timeout=20)


def _lookup_sid(account: str) -> str | None:
    """Resolve an account name to its SID string (S-1-5-...)."""
    try:
        import win32security  # type: ignore[import-not-found]
        sid, _, _ = win32security.LookupAccountName(None, account)
        return win32security.ConvertSidToStringSid(sid)
    except Exception:
        # Fall back to PowerShell so this works without pywin32.
        import subprocess
        ps = (
            f"(New-Object System.Security.Principal.NTAccount('{account}'))"
            ".Translate([System.Security.Principal.SecurityIdentifier]).Value"
        )
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, check=False,
                           timeout=20)
        out = (r.stdout or "").strip()
        return out if out.startswith("S-1-") else None


__all__ = [
    "install_launchd", "install_launchdaemon", "install_systemd",
    "install_windows_task", "install_xdg_autostart",
    "grant_windows_service_control", "LAUNCHDAEMON_DIR",
]
