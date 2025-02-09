import logging
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

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


def get_cpu_info() -> tuple[str, str, int]:
    """
    Retrieve a semi human-friendly CPU model name, technical CPU details,
    and the number of physical CPU cores.

    :return: A tuple (model_name, technical_name, core_count)
    """
    import psutil

    system = platform.system()

    def get_core_count() -> int:
        core_count = psutil.cpu_count(logical=False)
        return core_count if core_count else 1

    if system == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            m = re.search(r"model name\s*:\s*(.*)", cpuinfo)
            model_name = m.group(1).strip() if m else ""
        except Exception:
            model_name = ""
        technical_name = platform.processor() or "Unavailable"
        return (model_name, technical_name, get_core_count())
    elif system == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "Name"],
                check=True,
                capture_output=True,
                text=True,
            )
            lines = result.stdout.strip().splitlines()
            model_name = lines[1].strip() if len(lines) >= 2 else ""
        except Exception:
            model_name = ""
        technical_name = platform.processor() or "Unavailable"
        return (model_name, technical_name, get_core_count())
    else:
        technical_name = platform.processor() or "Unavailable"
        return ("", technical_name, get_core_count())


def get_hardware_info() -> dict[str, Any]:
    """
    Gather hardware information including total RAM, CPU technical information,
    and the number of CPU cores.
    """
    # (model_name, technical_name, core_count)
    _, technical_name, core_count = get_cpu_info()
    try:
        total_ram = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        total_ram = "Unavailable"
    return {
        "CPU": technical_name,
        "CoreCount": core_count,
        "TotalRAMGB": total_ram,
    }


def query_nvidia_gpu_extended_info(index: str) -> str:
    """
    Query limited extended information for NVIDIA GPU using nvidia-smi.
    Only returns the driver version and power limit.

    :param index: GPU index as a string.
    :return: A formatted string with NVIDIA GPU driver version and power limit.
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
    Retrieve Git information including current branch, commit hash,
    and any modified files compared to the upstream branch.

    :return: A formatted string with Git details or an error message if not available.
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

    :param url: The URL to test.
    :return: A string describing the success or failure and packet loss.
    """
    logging.info("Pinging URL: %s", url)
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.netloc
    count = "4"  # number of ping packets

    # Determine ping parameters based on OS
    cmd = ["ping", "-n", count, host] if platform.system() == "Windows" else ["ping", "-c", count, host]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
        output = result.stdout
        packet_loss = "Unavailable"
        if platform.system() == "Windows":
            # Windows output sample: "Lost = 0 (0% loss)"
            m = re.search(r"\((\d+)% loss\)", output)
            if m:
                packet_loss = m.group(1) + "%"
        else:
            # Linux output sample: "0.0% packet loss"
            m = re.search(r"(\d+(?:\.\d+)?)% packet loss", output)
            if m:
                packet_loss = m.group(1) + "%"
        return f"Ping to {host} successful: Packet Loss: {packet_loss}"
    except subprocess.CalledProcessError as e:
        logging.warning(
            "Ping to %s failed. Using requests fallback. Error: %s",
            host,
            e,
        )
        try:
            import requests

            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return f"Requests to {host} succeeded (fallback)."
            else:
                return f"Requests to {host} failed (fallback). Status code: {r.status_code}"
        except Exception as ex:
            return f"Requests fallback also failed: {ex}"
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
    report.append(
        f"CPU: {hardware_info['CPU']} (Cores: {hardware_info['CoreCount']})"
    )
    report.append(f"Total RAM: {hardware_info['TotalRAMGB']} GB")
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
    report.append(f"Current Git {git_branch}")
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
