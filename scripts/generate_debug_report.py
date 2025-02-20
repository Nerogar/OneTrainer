#!/usr/bin/env python3
import json
import logging
import os
import platform
import re
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import psutil
import requests

# Generating a debug report:
# Mac/Linux: Execute `./run-cmd.sh generate_debug_report`
# Windows: Double-click on `export_debug.bat`

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# == Helper Functions ==


def subprocess_run(
    cmd: list[str], **kwargs
) -> subprocess.CompletedProcess:
    """
    Run a subprocess with enforced locale settings and default kwargs.
    """
    proc_env = os.environ.copy()
    # Force external utilities to output US English ASCII with US formatting
    proc_env["LC_ALL"] = "C"
    kwargs.setdefault("check", True)
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    kwargs["env"] = proc_env
    return subprocess.run(cmd, **kwargs)


def run_command(cmd: list[str]) -> str | None:
    """
    Run a shell command using subprocess_run.
    Returns the stripped stdout if successful; otherwise, None.
    """
    try:
        result = subprocess_run(cmd)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(
            f'Command "{" ".join(map(shlex.quote, cmd))}" failed: {e.stderr}'
        )
    except Exception:
        logger.exception(
            f'Unexpected error running command: "{" ".join(map(shlex.quote, cmd))}"'
        )
    return None


def safe_call(func, default, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.exception(f"Error in {func.__name__}: {e}")
        return default


def anonymize_path(path: str | None) -> str | None:
    """
    Anonymize user paths for both Windows, Linux and Mac as there is no need to collect that.
    """
    if not path:
        return path
    # Replace Windows user paths.
    path = re.sub(r"(?i)^([A-Z]:\\Users)\\[^\\]+", r"\1\\anonymous", path)
    # Replace Linux user paths.
    path = re.sub(r"(?i)^/home/[^/]+", r"/home/anonymous", path)
    # Replace MacOS user paths.
    path = re.sub(r"(?i)^/Users/[^/]+", r"/Users/anonymous", path)
    return path


# == OS and CPU Functions ==


def get_os_info() -> dict[str, Any]:
    """
    Return a dictionary with OS details.
    """
    try:
        uname = platform.uname()
        system = uname.system
        version = uname.version

        # Special handling for Linux distributions
        if system == "Linux":
            # Try to get distribution info using /etc/os-release
            try:
                with open("/etc/os-release", "r") as f:
                    os_release = {}
                    for line in f:
                        if "=" in line:
                            key, value = line.rstrip().split("=", 1)
                            os_release[key] = value.strip('"')

                # Use PRETTY_NAME or fallback to NAME + VERSION_ID
                if "PRETTY_NAME" in os_release:
                    version = os_release["PRETTY_NAME"]
                elif "NAME" in os_release and "VERSION_ID" in os_release:
                    version = (
                        f"{os_release['NAME']} {os_release['VERSION_ID']}"
                    )
            except Exception:
                # Fallback to lsb_release command if available
                try:
                    result = subprocess.run(
                        ["lsb_release", "-d"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    if result.stdout.startswith("Description:"):
                        version = result.stdout.split(":", 1)[1].strip()
                except Exception:
                    # Keep original version if both methods fail
                    pass

        return {
            "System": system,
            "Node": uname.node,
            "Release": uname.release,
            "Version": version,
            "Machine": uname.machine,
            "Processor": uname.processor,
        }
    except Exception as e:
        logger.exception("Failed to get OS info")
        return {"Error": str(e)}


def get_cpu_info() -> tuple[str, str, int]:
    """
    Retrieve a semi human-friendly CPU model name, technical CPU details,
    and the number of physical CPU cores.

    Returns:
        A tuple (model_name, technical_name, core_count)
    """
    system = platform.system()

    def get_core_count() -> int:
        return psutil.cpu_count(logical=False) or 1

    if system == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            m = re.search(r"model name\s*:\s*(.*)", cpuinfo)
            model_name = m.group(1).strip() if m else "Unavailable"
        except Exception:
            logger.exception("Error reading /proc/cpuinfo")
            model_name = "Unavailable"
        technical_name = platform.processor() or "Unavailable"
        return model_name, technical_name, get_core_count()
    elif system == "Windows":
        try:
            result = subprocess_run(["wmic", "cpu", "get", "Name"])
            # Split lines, filter empty ones, skip header
            lines = [
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip()
            ]
            model_name = lines[1] if len(lines) > 1 else "Unavailable"
        except Exception:
            logger.exception("Error retrieving CPU info via WMIC")
            model_name = "Unavailable"
        technical_name = platform.processor() or "Unavailable"
        return model_name, technical_name, get_core_count()
    elif system == "Darwin":
        try:
            result = subprocess_run(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            )
            model_name = result.stdout.strip() or "Unavailable"
        except Exception:
            logger.exception("Error retrieving CPU info via sysctl")
            model_name = "Unavailable"
        # Use platform.machine() as fallback for the technical name
        technical_name = (
            platform.processor() or platform.machine() or "Unavailable"
        )
        # (Optional) Log if Apple Silicon is detected:
        if platform.machine() == "arm64":
            logger.info("Apple Silicon (arm64) detected.")
        return model_name, technical_name, get_core_count()
    else:
        model_name = "Unavailable"
        technical_name = platform.processor() or "Unavailable"
        return model_name, technical_name, get_core_count()


def get_system_specifications() -> dict[str, Any]:
    """
    Gather system hardware specifications including CPU details and total RAM.
    """
    model_name, technical_name, core_count = get_cpu_info()
    try:
        total_ram = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        logger.exception("Error retrieving RAM info")
        total_ram = "Unavailable"
    return {
        "CPU": model_name,
        "TechnicalCPU": technical_name,
        "CoreCount": core_count,
        "TotalRAMGB": total_ram,
    }


def get_intel_microcode_info() -> str:
    """
    Retrieve microcode information for Intel CPUs on supported systems.
    """
    result = "CPU is not detected as 13th or 14th Gen Intel - microcode info not applicable."
    try:
        cpu_name = platform.processor() or ""
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            m = re.search(r"microcode\s*:\s*(\S+)", cpuinfo)
            microcode = m.group(1) if m else "Unavailable"
            if re.search(r"i\d-13", cpu_name, re.IGNORECASE):
                result = (
                    f"13th Gen detected. Microcode revision: {microcode}"
                )
            elif re.search(r"i\d-14", cpu_name, re.IGNORECASE):
                result = (
                    f"14th Gen detected. Microcode revision: {microcode}"
                )
        else:
            if re.search(r"i\d-13", cpu_name, re.IGNORECASE):
                result = (
                    "13th Gen detected. Microcode revision: Unavailable"
                )
            elif re.search(r"i\d-14", cpu_name, re.IGNORECASE):
                result = (
                    "14th Gen detected. Microcode revision: Unavailable"
                )
    except Exception as e:
        logger.exception("Error retrieving microcode information")
        result = f"Unable to retrieve microcode information: {e}"
    return result


# == GPU Functions ==


def determine_vendor(vendor_str: str) -> str:
    lower_vendor = vendor_str.lower()
    if any(
        x.lower() in lower_vendor for x in ["AMD", "ATI", "Advanced Micro"]
    ):
        return "AMD"
    elif "intel" in lower_vendor:
        return "Intel"
    elif "nvidia" in lower_vendor:
        return "NVIDIA"
    return "Unknown"


def build_extended_info(info_items: list[str]) -> str:
    return "\n".join("    " + item for item in info_items) if info_items else "No extended info available"


def get_nvidia_gpu_info() -> list[dict[str, Any]]:
    output = run_command(["nvidia-smi", "-L"])
    if not output:
        logger.error("nvidia-smi did not return valid output.")
        return []
    nvidia_gpus = [
        {"Index": m.group(1), "Name": m.group(2)}
        for line in output.splitlines()
        if (m := re.match(r"GPU (\d+): (.*) \(UUID:", line))
    ]
    return [
        {
            "Identifier": f"NVIDIA GPU (Index {gpu['Index']})",
            "Name": gpu["Name"],
            "ExtendedInfo": query_nvidia_gpu_extended_info(gpu["Index"]),
            "Vendor": "NVIDIA",
        }
        for gpu in nvidia_gpus
    ]


def query_nvidia_gpu_extended_info(index: str) -> str:
    """
    Query extended information for NVIDIA GPU using nvidia-smi.
    Returns the driver version and power limit.
    """
    query = "driver_version,power.limit"
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
        "-i",
        index,
    ]
    merged_output = run_command(cmd)
    if not merged_output or "NVIDIA-SMI has failed" in merged_output:
        return "    Extended info unavailable."
    values = [val.strip() for val in merged_output.split(",")]
    if len(values) < 2:
        return "    Extended info unavailable."
    info_lines = [
        f"    Driver version: {values[0]}",
        f"    Power Limit: {values[1]} W",
    ]
    return "\n".join(info_lines)


def get_lshw_gpu_info() -> dict[str, Any]:
    """
    Parse GPU information from lshw JSON output using filtering to display class only.
    """
    try:
        # Run lshw with JSON output
        result = subprocess_run(
            ["lshw", "-json", "-sanitize", "-C", "display"]
        )
        devices = json.loads(result.stdout)

        # Handle both single device and list of devices
        if not isinstance(devices, list):
            devices = [devices]

        gpu = {
            "Identifier": "Unknown",
            "Name": "Unknown",
            "ExtendedInfo": "",
            "Vendor": "Unknown",
        }

        for device in devices:
            if device.get("class") == "display":
                # Get device identifier from PCI bus info
                gpu["Identifier"] = device.get("businfo", "Unknown")

                # Construct name from product info
                name_parts = []
                if device.get("product"):
                    name_parts.append(device["product"])
                if device.get("description"):
                    name_parts.append(device["description"])
                gpu["Name"] = (
                    " ".join(name_parts) if name_parts else "Unknown"
                )

                # Determine vendor using the new helper function
                vendor_str = device.get("vendor", "")
                gpu["Vendor"] = determine_vendor(vendor_str)

                # Build extended info using the new helper function
                extended = []
                if device.get("version"):
                    extended.append(f"Version: {device['version']}")
                if device.get("configuration", {}).get("driver"):
                    extended.append(
                        f"Driver: {device['configuration']['driver']}"
                    )
                if device.get("capabilities") and isinstance(
                    device["capabilities"], dict
                ):
                    extended.append(
                        "Capabilities: "
                        + ", ".join(device["capabilities"].keys())
                    )
                gpu["ExtendedInfo"] = build_extended_info(extended)

                break  # Take first display device

        return gpu

    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing lshw JSON output: {e}")
        return {
            "Identifier": "Error",
            "Name": "Error parsing GPU info",
            "ExtendedInfo": str(e),
            "Vendor": "Unknown",
        }


def get_lspci_gpu_info() -> list[dict[str, Any]]:
    """
    Retrieve GPU information using lspci.
    Parses lspci output and returns a list of non-NVIDIA GPUs.
    """
    RE_CLASS = re.compile(
        r"^Class:\s*(VGA compatible controller|3D controller)",
        re.MULTILINE,
    )
    RE_VENDOR = re.compile(r"^Vendor:\s*(.+)$", re.MULTILINE)
    RE_DEVICE = re.compile(r"^Device:\s*(.+)$", re.MULTILINE)
    RE_SLOT = re.compile(r"^\s*(.+)$", re.MULTILINE)

    gpus: list[dict[str, Any]] = []
    try:
        output = run_command(["lspci", "-vmm"])
    except Exception:
        logger.exception("Error running lspci")
        return gpus

    if not output:
        return gpus

    # Split into device blocks - each starts with "Slot:"
    device_blocks = re.split(r"(?m)^Slot:", output)
    for block in device_blocks:
        if not block.strip():
            continue
        # Check if this is a VGA/display device using the compiled regex
        if not RE_CLASS.search(block):
            continue
        vendor_match = RE_VENDOR.search(block)
        device_match = RE_DEVICE.search(block)
        slot_match = RE_SLOT.search(block)
        if not (vendor_match and device_match and slot_match):
            continue

        vendor_str = vendor_match.group(1).strip()
        # Skip NVIDIA GPUs since they are handled elsewhere
        if determine_vendor(vendor_str) == "NVIDIA":
            continue

        slot = slot_match.group(1).strip()
        gpus.append(
            {
                "Identifier": f"PCI Slot {slot}",
                "Name": device_match.group(1).strip(),
                "Vendor": determine_vendor(vendor_str),
                "ExtendedInfo": build_extended_info(
                    [
                        f"Vendor: {vendor_str}",
                        f"PCI Slot: {slot}",
                    ]
                ),
            }
        )
    return gpus


def get_generic_gpu_info() -> list[dict[str, Any]]:
    """
    Retrieve driver-specific GPU info for Intel/AMD on Windows via PowerShell.
    Returns a list of GPU info dictionaries.
    """
    ps_command = textwrap.dedent(r"""
        Get-CimInstance -ClassName Win32_VideoController | ForEach-Object {
            $adapterRAM = if ($_.AdapterRAM -gt 0) { $_.AdapterRAM } else {
                if ($_.VideoModeDescription -match '(\d+) x (\d+) x (\d+) colors') {
                    $width = [int]$Matches[1]
                    $height = [int]$Matches[2]
                    $colors = [int]$Matches[3]
                    $bitsPerPixel = [Math]::Log($colors, 2)
                    $estimatedRAM = ($width * $height * $bitsPerPixel / 8)
                    [Math]::Max($estimatedRAM, 1GB)
                } else { 0 }
            }
            [PSCustomObject]@{
                Name = $_.Name
                AdapterCompatibility = $_.AdapterCompatibility
                DriverVersion = $_.DriverVersion
            }
        } | Where-Object {
            $_.AdapterCompatibility -match 'NVIDIA|AMD|ATI|Advanced Micro Devices|Intel' -and
            $_.Name -notmatch 'Intel.*Graphics' -and
            $_.Name -notmatch 'AMD.*Graphics$' -and (
                $_.Name -match 'NVIDIA|GeForce|Quadro|Tesla' -or
                $_.Name -match 'AMD|Radeon|FirePro' -or
                $_.Name -match 'Intel.*Arc'
            )
        } | ConvertTo-Json
    """)
    try:
        result = subprocess_run(["powershell", "-Command", ps_command])
        output = result.stdout.strip()
        if output:
            parsed = json.loads(output)
            gpu_list = [parsed] if isinstance(parsed, dict) else parsed
            for gpu in gpu_list:
                gpu.setdefault(
                    "Identifier", gpu.get("Name", "Unknown GPU")
                )
                gpu.setdefault(
                    "Vendor", gpu.get("AdapterCompatibility", "Unknown")
                )
            return gpu_list
    except Exception:
        logger.exception("Error retrieving Intel/AMD GPU info")
    return []


def get_macos_gpu_info() -> list[dict[str, Any]]:
    """
    Retrieve GPU information on macOS using system_profiler.
    Returns a list of dictionaries containing GPU details.
    """
    gpus: list[dict[str, Any]] = []
    try:
        output = run_command(
            ["system_profiler", "SPDisplaysDataType", "-json"]
        )
        if output:
            data = json.loads(output)
            displays = data.get("SPDisplaysDataType", [])
            for display in displays:
                gpu_name = display.get("sppci_model", "Unknown")
                vendor = display.get("spdisplays_vendor", "Unknown")
                identifier = display.get("spdisplays_bus", "Unknown")
                extended_info_parts = []
                if "spdisplays_vram" in display:
                    extended_info_parts.append(
                        f"VRAM: {display['spdisplays_vram']}"
                    )
                if "spdisplays_resolution" in display:
                    extended_info_parts.append(
                        f"Resolution: {display['spdisplays_resolution']}"
                    )
                extended_info = build_extended_info(extended_info_parts)

                gpus.append(
                    {
                        "Identifier": identifier,
                        "Name": gpu_name,
                        "Vendor": vendor,
                        "ExtendedInfo": extended_info,
                    }
                )
    except Exception:
        logger.exception("Error retrieving GPU info on macOS")
    return gpus


def get_gpu_info() -> list[dict[str, Any]]:
    """
    Gather GPU information using a multi-stage approach.
    First, try NVIDIA native detection; if that fails,
    then attempt to use lspci; if that also fails,
    fall back to parsing lshw JSON output.
    Each branch is wrapped in try/except blocks for error tolerance.
    """
    try:
        system = platform.system()

        # x86 MacOS (Old and Hackintoshes) are not supported. M-series macs are all thats accounted for.
        if system == "Darwin":
            return safe_call(get_macos_gpu_info, [])

        # Obviously NVIDIA GPUs are the most common, so we check for them first after short-circuiting for Mac.
        # Approach works for both Windows and Linux.
        nvidia = safe_call(get_nvidia_gpu_info, [])
        if nvidia:
            return nvidia

        # Next most common case is Intel/AMD GPUs on Windows, check via powershell
        if system == "Windows":
            return safe_call(get_generic_gpu_info, [])

        # If not nvidia and is Linux, try lspci first, then fall back to lshw.
        if system == "Linux":
            lspci = safe_call(get_lspci_gpu_info, [])
            if lspci:
                return lspci
            lshw = safe_call(get_lshw_gpu_info, {})
            if lshw.get("Name") and lshw.get("Name") != "Unknown":
                return [lshw]

        return []
    except Exception as e:
        logger.exception(f"Error in get_gpu_info: {e}")
        return [
            {
                "Identifier": "Unknown",
                "Name": "Unknown GPU",
                "Vendor": "Unknown",
                "ExtendedInfo": build_extended_info([]),
            }
        ]


# == Software Functions ==


def get_python_info() -> dict[str, Any]:
    """
    Retrieve Python environment information including version, executable path,
    pip freeze output, and PyTorch info.
    """
    info: dict[str, Any] = {
        "PythonVersion": sys.version.split()[0],
        "PythonPath": sys.executable,
        "PipFreeze": "Unavailable",
        "PyTorchInfo": "PyTorch not detected",
    }
    cmd = [sys.executable, "-m", "pip", "freeze"]
    pip_out = run_command(cmd)
    if pip_out:
        info["PipFreeze"] = "\n".join(
            f"    {line}" for line in pip_out.splitlines()
        )
        for line in pip_out.splitlines():
            if line.startswith("torch=="):
                info["PyTorchInfo"] = line.strip()
    else:
        info["PipFreeze"] = "Unable to run pip freeze"
    return info


def get_git_info() -> str:
    """
    Retrieve Git information including current branch, commit hash,
    and any modified files compared to the upstream branch.
    """
    branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branch is None:
        return "Not a Git repository or git not installed."

    commit = run_command(["git", "rev-parse", "HEAD"]) or "Unavailable"
    git_info = f"Branch: {branch}\nCommit: {commit}"

    upstream = run_command(
        [
            "git",
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{u}",
        ]
    )
    if upstream:
        diff_files = (
            run_command(["git", "diff", "--name-only", "@{u}"]) or ""
        )
        if diff_files.strip():
            modified_files = "\n".join(
                f"  {line}" for line in diff_files.splitlines()
            )
            git_info += f"\nModified Files (differs from {upstream}):\n{modified_files}"
        else:
            git_info += (
                f"\nNo modifications relative to upstream ({upstream})."
            )
    else:
        git_info += "\nNo upstream branch tracking information available."

    return git_info


# == Miscellaneous Functions ==


def test_url(url: str) -> str:
    """
    Test network connectivity by pinging the host extracted from a URL.
    Reports packet loss from the ping command.
    """
    logger.info(f"Pinging URL: {url}")
    parsed = urlparse(url)
    host = parsed.netloc
    count = "4"  # Number of ping packets

    cmd = (
        ["ping", "-n", count, host]
        if platform.system() == "Windows"
        else ["ping", "-c", count, host]
    )
    try:
        result = subprocess_run(cmd)
        output = result.stdout
        packet_loss = "Unavailable"
        if platform.system() == "Windows":
            m = re.search(r"\((\d+)% loss\)", output)
            if m:
                packet_loss = f"{m.group(1)}%"
        else:
            m = re.search(r"(\d+(?:\.\d+)?)% packet loss", output)
            if m:
                packet_loss = f"{m.group(1)}%"
        return f"Ping to {host} successful: Packet Loss: {packet_loss}"
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"Ping to {host} failed. Using requests fallback. Error: {e}"
        )
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return f"Requests to {host} succeeded (fallback)."
            else:
                return f"Requests to {host} failed (fallback). Status code: {r.status_code}"
        except Exception as ex:
            logger.exception("Requests fallback failed")
            return f"Requests fallback also failed: {ex}"
    except Exception as e:
        logger.exception("Unexpected error during ping")
        return f"Failure: {e}"


# == Main Functions ==


def build_report() -> list[str]:
    """
    Collect system information and build the debug report.
    """
    os_info = get_os_info()
    hardware_info = get_system_specifications()
    gpu_info = get_gpu_info()
    python_info = get_python_info()
    git_info = get_git_info()
    pyPi_status = test_url("https://pypi.org/")
    huggingface_status = test_url("https://huggingface.co")
    google_status = test_url("https://www.google.com")
    intel_microcode = get_intel_microcode_info()

    report = [
        "=== System Information ===",
        f"OS: {os_info.get('System', 'Unavailable')} {os_info.get('Release', '')}",
        f"Version: {os_info.get('Version', 'Unavailable')}",
        "",
        "=== Hardware Information ===",
        f"CPU: {hardware_info.get('CPU', 'Unavailable')} (Cores: {hardware_info.get('CoreCount', 'Unavailable')})",
        f"Total RAM: {hardware_info.get('TotalRAMGB', 'Unavailable')} GB",
        "",
        "=== GPU Information ===",
    ]
    if gpu_info:
        for gpu in gpu_info:
            report.append(
                f"{gpu.get('Identifier', 'Unknown')}: {gpu.get('Name', 'Unnamed')} [{gpu.get('Vendor', 'Unknown')}]"
            )
            extended_info = gpu.get("ExtendedInfo", "")
            if extended_info:
                report.append(extended_info)
    else:
        report.append("No GPUs detected.")
    report.extend(
        [
            "",
            "=== Python Environment ===",
            f"Global Python Version: {python_info.get('PythonVersion', 'Unavailable')}",
            f"Python Executable Path: {anonymize_path(python_info.get('PythonPath'))}",
            f"PyTorch Info: {python_info.get('PyTorchInfo', 'Unavailable')}",
            "pip freeze output:",
            python_info.get("PipFreeze", "Unavailable"),
            "",
            "=== Git Information ===",
            git_info,
            "",
            "=== Network Connectivity ===",
            f"PyPI (https://pypi.org/): {pyPi_status}",
            f"HuggingFace (https://huggingface.co): {huggingface_status}",
            f"Google (https://www.google.com): {google_status}",
            "",
            "=== Intel Microcode Information ===",
            intel_microcode,
            "",
        ]
    )
    return report


def write_report(report: list[str], output_file: Path) -> None:
    """
    Write the report to the given output file with final anonymization.
    """
    try:
        anonymized_report = "\n".join(
            anonymize_path(line) for line in report
        )
        output_file.write_text(anonymized_report, encoding="utf-8")
        logger.info(f"Report assembled and saved to {output_file}")
    except Exception:
        logger.exception("Failed to write report to file")


def main() -> None:
    """
    Main function to collect info, build the debug report, and write it to a file.
    """
    current_dir = Path.cwd()
    logger.info(f"Current directory: {current_dir}")
    if current_dir.name != "OneTrainer":
        logger.warning(
            f"Expected to run from the OneTrainer folder. Current folder: {current_dir}"
        )

    report = build_report()
    output_file = Path("debug_report.log")
    write_report(report, output_file)


if __name__ == "__main__":
    main()
