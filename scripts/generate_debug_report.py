import logging
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

try:
    import psutil
except ImportError:
    logging.error(
        "psutil module is required. Install it with: pip install psutil"
    )
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def run_command(cmd: list[str]) -> str | None:
    """
    Helper function to run a shell command using subprocess.run.

    :param cmd: The command and arguments to run.
    :return: The standard output as a string if successful, otherwise None.
    """
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(
            "Command '%s' failed with error: %s", " ".join(cmd), e.stderr
        )
    except Exception as e:
        logging.error(
            "Unexpected error when running command '%s': %s",
            " ".join(cmd),
            e,
        )
    return None


def convert_path(path: str | None) -> str | None:
    """
    Anonymize user paths for both Windows and Linux.

    :param path: The original path string.
    :return: The anonymized path.
    """
    if not path:
        return path
    # Replace Windows user paths
    path = re.sub(r"C:\\Users\\[^\\]+", r"C:\\Users\\anonymous", path)
    # Replace Linux user paths
    path = re.sub(r"/home/[^/]+", r"/home/anonymous", path)
    return path


def get_os_info() -> dict[str, Any]:
    """Return a dictionary with OS details."""
    try:
        uname = platform.uname()
        os_info = {
            "System": uname.system,
            "Node": uname.node,
            "Release": uname.release,
            "Version": uname.version,
            "Machine": uname.machine,
            "Processor": uname.processor,
        }
    except Exception as e:
        os_info = {"Error": str(e)}
    return os_info


def get_cpu_info() -> tuple[str, str]:
    """
    Retrieve a semi human-friendly CPU model name and the technical CPU details. This is not the marketing name.

    :return: A tuple (model_name, technical_name)
    """
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            m = re.search(r"model name\s*:\s*(.*)", cpuinfo)
            # Human-friendly model name; if not found, leave it empty.
            model_name = m.group(1).strip() if m else ""
        except Exception:
            model_name = ""
        technical_name = platform.processor() or "Unavailable"
        return (model_name, technical_name)
    elif system == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "Name"],
                check=True,
                capture_output=True,
                text=True,
            )
            lines = result.stdout.strip().splitlines()
            # Take the WMIC output if available, else return empty string.
            model_name = lines[1].strip() if len(lines) >= 2 else ""
        except Exception:
            model_name = ""
        technical_name = platform.processor() or "Unavailable"
        return (model_name, technical_name)
    else:
        technical_name = platform.processor() or "Unavailable"
        return ("", technical_name)


# TODO Figure out how to determine what type of drive it is; SATA or NVME


def get_hardware_info() -> dict[str, Any]:
    """
    Gather hardware information including total RAM, drive details,
    and CPU technical information.

    :return: Dictionary containing CPU technical details, total RAM in GB,
             and drive information.
    """
    _, technical_name = get_cpu_info()
    try:
        total_ram = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        total_ram = "Unavailable"
    drives: list[str] = []
    try:
        partitions = psutil.disk_partitions(all=False)
        for p in partitions:
            try:
                usage = psutil.disk_usage(p.mountpoint)
                size_gb = round(usage.total / (1024**3), 2)
            except Exception:
                size_gb = "Unknown"
            drives.append(
                f"Drive: {p.device} mounted on {p.mountpoint} | Size: {size_gb} GB | Type: {p.fstype}"
            )
    except Exception:
        drives.append("No physical drive information available.")
    return {
        "CPU": technical_name,
        "TotalRAMGB": total_ram,
        "Drives": drives,
    }


def query_nvidia_gpu_extended_info(index: str) -> str:
    """
    Query extended information for NVIDIA GPU using nvidia-smi.

    :param index: GPU index as a string.
    :return: A formatted string with extended NVIDIA GPU info.
    """
    query = (
        "driver_version,power.limit,clocks.current.graphics,clocks.current.memory,"
        "temperature.gpu,pstate,clocks_event_reasons.supported,clocks_event_reasons.active,"
        "clocks_event_reasons.gpu_idle,clocks_event_reasons.applications_clocks_setting,"
        "clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown,"
        "clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown,"
        "clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.sync_boost"
    )
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
        "-i",
        index,
    ]
    merged_output = run_command(cmd)
    if not merged_output:
        return "    Extended info unavailable."
    if (
        "NVIDIA-SMI has failed" in merged_output
        or "NotSupported" in merged_output
    ):
        return "    Extended info unavailable."
    values = [val.strip() for val in merged_output.split(",")]
    info_lines = [
        f"    Driver version: {values[0]}",
        f"    Power Limit: {values[1]} W",
        f"    Current core clock: {values[2]} MHz",
        f"    Memory clock: {values[3]} MHz",
        f"    Temp: {values[4]} °C",
        f"    PState: {values[5]}",
        "    Clocks Throttle Reasons:",
        f"         Supported: {values[6]}",
        f"         Active: {values[7]}",
        f"         GPU Idle: {values[8]}",
        f"         Applications Clocks Setting: {values[9]}",
        f"         SW Power Cap: {values[10]}",
        f"         HW Slowdown: {values[11]}",
        f"         HW Thermal Slowdown: {values[12]}",
        f"         HW Power Brake Slowdown: {values[13]}",
        f"         SW Thermal Slowdown: {values[14]}",
        f"         Sync Boost: {values[15]}",
    ]
    return "\n".join(info_lines)


def get_nvidia_gpu_info() -> list[dict[str, Any]]:
    """
    Retrieve NVIDIA GPU information using nvidia-smi.

    :return: A list of dictionaries containing NVIDIA GPU details.
    """
    nvidia_gpus: list[dict[str, str]] = []
    cmd = ["nvidia-smi", "-L"]
    output = run_command(cmd)
    if not output:
        logging.error("nvidia-smi did not return valid output.")
        return []
    for line in output.splitlines():
        m = re.match(r"GPU (\d+): (.*) \(UUID:", line)
        if m:
            index = m.group(1)
            name = m.group(2)
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


def get_generic_gpu_info() -> list[dict[str, Any]]:
    """
    Retrieve GPU information from non-NVIDIA sources such as AMD or Intel systems using lspci.

    :return: A list of dictionaries containing GPU details.
    """
    gpus: list[dict[str, Any]] = []
    if platform.system() == "Windows":
        # Limited support on Windows without additional modules.
        return gpus
    cmd = ["lspci"]
    output = run_command(cmd)
    if output:
        for line in output.splitlines():
            if re.search("VGA compatible controller", line, re.IGNORECASE):
                identifier = line.split()[0]
                name = " ".join(line.split()[1:])
                vendor = "Unknown"
                # Determine vendor based on keywords.
                if "AMD" in line or "ATI" in line:
                    vendor = "AMD"
                elif "Intel" in line:
                    vendor = "Intel"
                elif "NVIDIA" in line:
                    # Already handled in NVIDIA specific function.
                    continue
                gpus.append(
                    {
                        "Identifier": identifier,
                        "Name": name,
                        "ExtendedInfo": "    No extended info available.",
                        "Vendor": vendor,
                    }
                )
    return gpus


def get_gpu_info() -> list[dict[str, Any]]:
    """
    Gather GPU information from NVIDIA and generic sources (e.g., AMD, Intel).

    :return: A list of GPU information dictionaries.
    """
    gpus = get_nvidia_gpu_info()
    gpus += get_generic_gpu_info()
    return gpus


def get_python_info() -> dict[str, Any]:
    """
    Retrieve Python environment information including version, executable path, pip freeze output, and PyTorch info.

    :return: A dictionary with Python environment details.
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
            "    " + line for line in pip_out.splitlines()
        )
        for line in pip_out.splitlines():
            if line.startswith("torch=="):
                info["PyTorchInfo"] = line.strip()
    else:
        info["PipFreeze"] = "Unable to run pip freeze"
    return info


def get_git_info() -> str:
    """
    Retrieve the current Git branch name.

    :return: The current Git branch or an error message if not available.
    """
    cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    branch = run_command(cmd)
    if branch is None:
        return "Not a Git repository or git not installed."
    return branch


def test_url(url: str) -> str:
    """
    Test network connectivity by sending a HEAD request to a URL.

    :param url: The URL to test.
    :return: A string describing the success or failure of the request.
    """
    logging.info("Pinging URL: %s", url)
    try:
        start = datetime.now()
        response = requests.head(url, timeout=15)
        duration = int((datetime.now() - start).total_seconds() * 1000)
        if response.status_code == 200:
            return f"Success (Response Time: {duration} ms)"
        else:
            return f"Success (Status Code: {response.status_code}, Response Time: {duration} ms)"
    except Exception as e:
        return f"Failure: {e}"


def get_intel_microcode_info() -> str:
    """
    Retrieve microcode information for Intel CPUs on supported systems.

    :return: A string with microcode details or a message if not applicable.
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
        result = f"Unable to retrieve microcode information: {e}"
    return result


def main() -> None:
    """
    Main function to generate and save the debug report.
    """
    current_dir = Path.cwd()
    logging.info("Current directory: %s", current_dir)
    if current_dir.name != "OneTrainer":
        logging.warning(
            "Expected to run from the OneTrainer folder. Current folder: %s",
            current_dir,
        )

    os_info = get_os_info()
    hardware_info = get_hardware_info()
    gpu_info = get_gpu_info()
    python_info = get_python_info()
    git_branch = get_git_info()
    pyPi_status = test_url("https://pypi.org/")
    huggingface_status = test_url("https://huggingface.co")
    google_status = test_url("https://www.google.com")
    intel_microcode = get_intel_microcode_info()

    report: list[str] = []
    report.append("=== System Information ===")
    report.append(
        f"OS: {os_info.get('System', 'Unavailable')} {os_info.get('Release', '')}"
    )
    report.append(f"Version: {os_info.get('Version', 'Unavailable')}")
    report.append("")

    report.append("=== Hardware Information ===")
    report.append(f"CPU: {hardware_info['CPU']}")
    report.append(f"Total RAM: {hardware_info['TotalRAMGB']} GB")
    report.append("=== Drive Information ===")
    report.extend(
        hardware_info["Drives"]
        if hardware_info["Drives"]
        else ["No physical drive information available."]
    )
    report.append("")

    report.append("=== GPU Information ===")
    if gpu_info:
        for gpu in gpu_info:
            report.append(
                f"{gpu['Identifier']}: {gpu['Name']} [{gpu['Vendor']}]"
            )
            report.append(gpu["ExtendedInfo"])
    else:
        report.append("No GPUs detected.")
    report.append("")

    report.append("=== Python Environment ===")
    report.append(f"Global Python Version: {python_info['PythonVersion']}")
    report.append(
        f"Python Executable Path: {convert_path(python_info['PythonPath'])}"
    )
    report.append(f"PyTorch Info: {python_info['PyTorchInfo']}")
    report.append("pip freeze output:")
    report.append(python_info["PipFreeze"])
    report.append("")

    report.append("=== Git Information ===")
    report.append(f"Current Git Branch: {git_branch}")
    report.append("")

    report.append("=== Network Connectivity ===")
    report.append(f"PyPI (https://pypi.org/): {pyPi_status}")
    report.append(
        f"HuggingFace (https://huggingface.co): {huggingface_status}"
    )
    report.append(f"Google (https://www.google.com): {google_status}")
    report.append("")

    report.append("=== Intel Microcode Information ===")
    report.append(intel_microcode)
    report.append("")

    output_file = Path("debug_report.log")
    try:
        # Final anonymization of Windows user paths in report
        output_file.write_text(
            "\n".join(convert_path(line) for line in report),
            encoding="utf-8",
        )
        logging.info("Report assembled and saved to %s", output_file)
    except Exception as e:
        logging.error("Failed to write report to file: %s", e)


if __name__ == "__main__":
    main()
