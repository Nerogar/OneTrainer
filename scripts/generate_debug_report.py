#!/usr/bin/env python3
import json
import logging
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import psutil
import requests

# Run ./run-cmd.sh generate_debug_report to generate the debug report on Linux or Mac. Windows users should double click export_debug.bat

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Avoids locale-related issues with subprocesses as mentioned by Johnny
def subprocess_run(
    cmd: list[str], **kwargs
) -> subprocess.CompletedProcess:
    """
    Run a subprocess with enforced locale settings and default kwargs.
    """
    proc_env = os.environ.copy()
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
            f"Command '{' '.join(map(shlex.quote, cmd))}' failed: {e.stderr}"
        )
    except Exception:
        logger.exception(
            f"Unexpected error running command: {' '.join(map(shlex.quote, cmd))}"
        )
    return None


def anonymize_path(path: str | None) -> str | None:
    """
    Anonymize user paths for both Windows and Linux as there is no need to collect that.
    """
    if not path:
        return path
    # Replace Windows user paths.
    path = re.sub(r"(?i)^([A-Z]:\\Users)\\[^\\]+", r"\1\\anonymous", path)
    # Replace Linux user paths.
    path = re.sub(r"(?i)^/home/[^/]+", r"/home/anonymous", path)
    return path


def get_os_info() -> dict[str, Any]:
    """
    Return a dictionary with OS details.
    """
    try:
        uname = platform.uname()
        return {
            "System": uname.system,
            "Node": uname.node,
            "Release": uname.release,
            "Version": uname.version,
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
            match = re.search(r"model name\s*:\s*(.*)", cpuinfo)
            model_name = match.group(1).strip() if match else "Unavailable"
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
            model_name = lines[1] if len(lines) > 1 else ""
        except Exception:
            logger.exception("Error retrieving CPU info via WMIC")
            model_name = ""
        technical_name = platform.processor() or "Unavailable"
        return model_name, technical_name, get_core_count()
    else:
        technical_name = platform.processor() or "Unavailable"
        return model_name, technical_name, get_core_count()


def get_hardware_info() -> dict[str, Any]:
    """
    Gather hardware information including total RAM,
    CPU technical information, and the number of CPU cores.
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


def get_nvidia_gpu_info() -> list[dict[str, Any]]:
    """
    Retrieve NVIDIA GPU information using nvidia-smi.
    Returns a list of dictionaries containing NVIDIA GPU details.
    """
    nvidia_gpus: list[dict[str, str]] = []
    cmd = ["nvidia-smi", "-L"]
    output = run_command(cmd)
    if not output:
        logger.error("nvidia-smi did not return valid output.")
        return []
    for line in output.splitlines():
        match = re.match(r"GPU (\d+): (.*) \(UUID:", line)
        if match:
            index = match.group(1)
            name = match.group(2)
            nvidia_gpus.append({"Index": index, "Name": name})
    extended_gpus: list[dict[str, Any]] = []
    for gpu in nvidia_gpus:
        info = query_nvidia_gpu_extended_info(gpu["Index"])
        extended_gpus.append(
            {
                "Identifier": f"NVIDIA GPU (Index {gpu['Index']})",
                "Name": gpu["Name"],
                "ExtendedInfo": info,
                "Vendor": "NVIDIA",
            }
        )
    return extended_gpus


def parse_lshw_block(block_lines: list[str]) -> dict[str, Any]:
    """
    Parse a block of lshw output and extract GPU details.
    Fallback in case lspci is not available.
    """
    gpu = {
        "Identifier": "Unknown",
        "Name": "Unknown",
        "ExtendedInfo": "",
        "Vendor": "Unknown",
    }
    extended_info_lines = []
    found_vga = False

    for line in block_lines:
        if "VGA compatible controller:" in line:
            found_vga = True
            # Try to extract identifier (e.g., "01:00.0").
            id_match = re.search(r"\b\d{2}:\d{2}\.\d\b", line)
            if id_match:
                gpu["Identifier"] = id_match.group()
            else:
                tokens = line.split()
                gpu["Identifier"] = tokens[1] if len(tokens) > 1 else "N/A"
            details = line.split("VGA compatible controller:", 1)[
                1
            ].strip()
            gpu["Name"] = details
            if "AMD" in details or "ATI" in details:
                gpu["Vendor"] = "AMD"
            elif "Intel" in details:
                gpu["Vendor"] = "Intel"
            elif "NVIDIA" in details:
                gpu["Vendor"] = "NVIDIA"
        else:
            extended_info_lines.append(line.strip())
    if extended_info_lines:
        gpu["ExtendedInfo"] = "\n    ".join(
            line for line in extended_info_lines if line
        )
    if not found_vga:
        description = ""
        product = ""
        bus_info = ""
        for line in block_lines:
            line = line.strip()
            if line.startswith("*-"):
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip().lower()
                val = val.strip()
                if key == "description":
                    description = val
                elif key == "product":
                    product = val
                elif key == "vendor":
                    gpu["Vendor"] = val
                elif key == "bus info":
                    bus_info = val
        if bus_info:
            gpu["Identifier"] = bus_info
        name_parts = []
        if description:
            name_parts.append(description)
        if product:
            name_parts.append(product)
        gpu["Name"] = " ".join(name_parts) if name_parts else gpu["Name"]
    return gpu


def get_intel_amd_gpu_info() -> list[dict[str, Any]]:
    """
    Retrieve driver-specific GPU info for Intel/AMD (and some NVIDIA discrete)
    on Windows via PowerShell.
    Returns a list of GPU info dictionaries.
    """
    ps_command = r"""
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
                """
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


def get_generic_gpu_info() -> list[dict[str, Any]]:
    """
    Retrieve GPU information from non-NVIDIA sources such as AMD or Intel.
    On Windows, attempts to retrieve driver-specific info using PowerShell.
    """
    gpus: list[dict[str, Any]] = []
    if platform.system() == "Windows":
        return get_intel_amd_gpu_info()

    try:
        output = run_command(["lspci"])
    except Exception:
        logger.exception("Error running lspci")
        output = ""

    if output:
        for line in output.splitlines():
            if re.search("VGA compatible controller", line, re.IGNORECASE):
                parts = line.split()
                identifier = parts[0]
                name = " ".join(parts[1:])
                vendor = "Unknown"
                if "AMD" in line or "ATI" in line:
                    vendor = "AMD"
                elif "Intel" in line:
                    vendor = "Intel"
                elif "NVIDIA" in line:
                    continue  # NVIDIA handled separately.
                gpus.append(
                    {
                        "Identifier": identifier,
                        "Name": name,
                        "ExtendedInfo": "    No extended info available.",
                        "Vendor": vendor,
                    }
                )
    else:
        try:
            fallback_output = run_command(
                ["sudo", "lshw", "-numeric", "-C", "display"]
            )
        except Exception:
            logger.exception("Error running lshw")
            fallback_output = ""

        if fallback_output:
            block_lines = []
            in_display_block = False
            for line in fallback_output.splitlines():
                if line.lstrip().startswith("*-display"):
                    if in_display_block and block_lines:
                        gpu = parse_lshw_block(block_lines)
                        if gpu:
                            gpus.append(gpu)
                        block_lines = []
                    in_display_block = True
                    block_lines.append(line)
                elif in_display_block:
                    if line and not line.startswith(" "):
                        gpu = parse_lshw_block(block_lines)
                        if gpu:
                            gpus.append(gpu)
                        in_display_block = False
                        block_lines = []
                    else:
                        block_lines.append(line)
            if in_display_block and block_lines:
                gpu = parse_lshw_block(block_lines)
                if gpu:
                    gpus.append(gpu)
    return gpus


def get_gpu_info() -> list[dict[str, Any]]:
    """
    Gather GPU information from NVIDIA and generic sources.
    If NVIDIA GPUs are detected via nvidia-smi, only return those.
    Otherwise, fall back to generic GPU detection.
    """
    nvidia_gpus = get_nvidia_gpu_info()
    if not nvidia_gpus:
        return get_generic_gpu_info()
    return nvidia_gpus


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


def build_report() -> list[str]:
    """
    Collect system information and build the debug report.
    """
    os_info = get_os_info()
    hardware_info = get_hardware_info()
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
