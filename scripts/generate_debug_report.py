#!/usr/bin/env python3
import json
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from deepdiff import DeepDiff
except ImportError:
    DeepDiff = None
    logging.warning("deepdiff is not installed. Some features will be unavailable. Install with: pip install deepdiff")

try:
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.enum.ModelType import ModelType
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.enum.ModelType import ModelType

import psutil
import requests

# Generating a debug report:
# Mac/Linux: Execute `./run-cmd.sh generate_debug_report`
# Windows: Double-click on `export_debug.bat`

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

# == Helper Classes ==

def anonymize_config_dict(config_item):
    """Recursively anonymize paths in the configuration dictionary."""
    if isinstance(config_item, dict):
        return {k: anonymize_config_dict(v) for k, v in config_item.items()}
    elif isinstance(config_item, list):
        return [anonymize_config_dict(i) for i in config_item]
    elif isinstance(config_item, str):
        return Utility.anonymize_path(config_item)
    else:
        return config_item

def create_debug_package(output_zip_path: str, config_json_string: str):
    """Generates a zip file containing an anonymized config and a debug report."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            config_dict = json.loads(config_json_string)
            anonymized_config = anonymize_config_dict(config_dict)
            config_path = temp_path / "config.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(anonymized_config, f, indent=4)

            generate_default_config_file(temp_path)
            default_config_path = temp_path / "default_settings.json"
            diff_path = temp_path / "config_diff.txt"

            if DeepDiff and default_config_path.exists():
                try:
                    with open(default_config_path, "r", encoding="utf-8") as f:
                        default_config_dict = json.load(f)

                    diff = DeepDiff(
                        default_config_dict,
                        anonymized_config,
                        ignore_order=True,
                        verbose_level=2,
                        exclude_regex_paths=(
                            r"root\['(?:optimizer_defaults|concepts|samples|sample_definition_file_name|concept_file_name)'\]"
                            r"|root\['embedding'\]\['uuid'\]",
                            r"root\['text_encoder(?:_[1-4])?'\]\['model_name'\]",
                            r"root\['unet?'\]\['model_name'\]",
                        ),
                        ignore_string_type_changes=True,
                        ignore_numeric_type_changes=True,
                    )

                    diff.pop('type_changes', None)

                    formatted_diff = format_diff_output(diff)

                    with open(diff_path, "w", encoding="utf-8") as f:
                        f.write(formatted_diff)
                    logger.info("Generated config diff file.")
                except Exception as e:
                    logger.error(f"Could not generate config diff: {e}")
                    with open(diff_path, "w", encoding="utf-8") as f:
                        f.write(f"Error generating diff: {e}")
            elif not DeepDiff:
                logger.warning("deepdiff is not installed. Skipping config diff generation. `pip install deepdiff`")


            script_path = Path(__file__)
            onetrainer_dir = script_path.parent.parent  # Go up from scripts/ to OneTrainer/

            result = subprocess.run(
                [sys.executable, str(script_path)],
                check=False,
                cwd=onetrainer_dir,
                capture_output=True,
                text=True
            )

            source_report = onetrainer_dir / "debug_report.log"
            report_path = temp_path / "debug_report.log"

            if source_report.exists():
                shutil.copy2(source_report, report_path)
                source_report.unlink()

            with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(config_path, arcname="config.json")
                if diff_path.exists():
                    zf.write(diff_path, arcname="config_diff.txt")
                if report_path.exists():
                    zf.write(report_path, arcname="debug_report.log")
                else:
                    error_msg = f"Failed to generate debug_report.log.\nReturn code: {result.returncode}"
                    if result.stderr:
                        error_msg += f"\nError output:\n{result.stderr}"
                    if result.stdout:
                        error_msg += f"\nStandard output:\n{result.stdout}"
                    zf.writestr("debug_report.log", error_msg)

        logger.info(f"Debug package saved to {Path(output_zip_path).name}")
    except Exception as e:
        logger.error(f"Error generating debug package: {e}")
        traceback.print_exc()

def format_diff_output(diff) -> str:
    """Format DeepDiff output in a more human-readable way."""
    lines = []

    # Handle values_changed
    if 'values_changed' in diff:
        for path, change in diff['values_changed'].items():
            # Extract the key from the path
            key = extract_key_from_path(path)
            old_value = change['old_value']
            new_value = change['new_value']
            lines.append(f"{key}: {old_value} ⇒ {new_value}")

    # Handle dictionary_item_added
    if 'dictionary_item_added' in diff:
        for path in diff['dictionary_item_added']:
            key = extract_key_from_path(path)
            value = diff['dictionary_item_added'][path]
            lines.append(f"{key}: [ADDED] {value}")

    # Handle dictionary_item_removed
    if 'dictionary_item_removed' in diff:
        for path in diff['dictionary_item_removed']:
            key = extract_key_from_path(path)
            value = diff['dictionary_item_removed'][path]
            lines.append(f"{key}: [REMOVED] {value}")

    # Handle iterable_item_added
    if 'iterable_item_added' in diff:
        for path, value in diff['iterable_item_added'].items():
            key = extract_key_from_path(path)
            lines.append(f"{key}: [ADDED] {value}")

    # Handle iterable_item_removed
    if 'iterable_item_removed' in diff:
        for path, value in diff['iterable_item_removed'].items():
            key = extract_key_from_path(path)
            lines.append(f"{key}: [REMOVED] {value}")

    return "\n".join(lines) if lines else "No differences found."

def extract_key_from_path(path: str) -> str:
    """Extract a clean key path from DeepDiff path notation."""
    # Remove 'root' and clean up the path
    # Example: "root['training_method']" -> "training_method"
    # Example: "root['unet']['train']" -> "unet.train"

    # Remove 'root' prefix
    clean_path = path.replace('root', '')

    # Remove brackets and quotes, split by ']['
    parts = clean_path.strip('[]').replace("'", "").replace('][', '.').split('.')

    # Filter out empty parts
    parts = [part for part in parts if part]

    return '.'.join(parts)

def generate_default_config_file(output_dir: Path):
    """Generates a JSON file with the default TrainConfig settings for SD1.5 Fine-Tune."""
    try:
        logger.info("Generating default settings file for SD1.5 Fine-Tune...")
        default_config = TrainConfig.default_values()

        # Set defaults for a standard SD1.5 Fine-Tune scenario (which is the default for TrainConfig)
        default_config.model_type = ModelType.STABLE_DIFFUSION_15
        default_config.training_method = 'FINE_TUNE' # Using string to avoid enum import issues here

        # Enable training for components used in SD1.5 fine-tuning
        default_config.unet.train = True
        default_config.text_encoder.train = True

        # Disable training for components not used in SD1.5
        default_config.text_encoder_2.train = False
        default_config.text_encoder_3.include = False
        default_config.text_encoder_3.train = False
        default_config.text_encoder_4.include = False
        default_config.text_encoder_4.train = False
        default_config.prior.train = False

        # Inject default optimizer (AdamW) settings into optimizer_defaults
        from modules.util.optimizer_util import change_optimizer
        default_opt_cfg = change_optimizer(default_config)

        default_config.optimizer = default_opt_cfg

        default_config.optimizer_defaults[str(default_opt_cfg.optimizer)] = default_opt_cfg

        # The to_settings_dict method provides a clean representation of the config
        default_config_dict = default_config.to_settings_dict(secrets=False)

        output_file = output_dir / "default_settings.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(default_config_dict, f, indent=4, sort_keys=False)
        logger.info(f"Default settings file saved to {output_file}")
    except Exception as e:
        logger.error(f"Could not generate default settings file: {e}")
        traceback.print_exc()

class Utility:
    @staticmethod
    def subprocess_run(
        cmd: list[str], **kwargs
    ) -> subprocess.CompletedProcess:
        """Run a subprocess with enforced locale settings and default kwargs."""
        proc_env = os.environ.copy()
        # Force external utilities to output US English ASCII with US formatting
        proc_env["LC_ALL"] = "C"

        # Someone I showed didnt like multiple kwargs.setdefault, changed to this
        defaults = {
            "check": True,
            "capture_output": True,
            "text": True,
            "env": proc_env,
        }
        return subprocess.run(cmd, **{**defaults, **kwargs})

    @staticmethod
    def run_command(cmd: list[str]) -> str | None:
        """
        Run a shell command using subprocess_run.
        Returns the stripped stdout if successful; otherwise, None.
        """
        try:
            result = Utility.subprocess_run(cmd)
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

    @staticmethod
    def safe_call(func, default, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Error in {func.__name__}: {e}")
            return default

    @staticmethod
    def anonymize_path(path: str | None) -> str | None:
        """Anonymize user paths for both Windows, Linux and Mac."""
        if not path:
            return path

        patterns = [
            (r"(?i)^([A-Z]:\\Users)\\[^\\]+", r"\1\\anonymous"),  # Windows
            (r"(?i)^/home/[^/]+", r"/home/anonymous"),  # Linux
            (r"(?i)^/Users/[^/]+", r"/Users/anonymous"),  # macOS
        ]

        for pattern, replacement in patterns:
            path = re.sub(pattern, replacement, path)
        return path


class OSInfo:
    @staticmethod
    def get_info() -> dict[str, Any]:
        """
        Return a dictionary with OS details.
        """
        try:
            uname = platform.uname()
            system = uname.system
            version = uname.version

            # Special handling for Linux distributions
            if system == "Linux":
                try:
                    with open("/etc/os-release", "r") as f:
                        os_release = {
                            key: value.strip('"')
                            for line in f
                            if "=" in line
                            for key, value in [line.rstrip().split("=", 1)]
                        }

                    # Use PRETTY_NAME or fallback to NAME + VERSION_ID
                    if "PRETTY_NAME" in os_release:
                        version = os_release["PRETTY_NAME"]
                    elif (
                        "NAME" in os_release and "VERSION_ID" in os_release
                    ):
                        version = f"{os_release['NAME']} {os_release['VERSION_ID']}"
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
                            version = result.stdout.split(":", 1)[
                                1
                            ].strip()
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


class CPUInfo:
    @staticmethod
    def get_info() -> tuple[str, str, int]:
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
                result = Utility.subprocess_run(
                    ["wmic", "cpu", "get", "Name"]
                )
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
                result = Utility.subprocess_run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"]
                )
                model_name = result.stdout.strip() or "Unavailable"
            except Exception:
                logger.exception("Error retrieving CPU info via sysctl")
                model_name = "Unavailable"
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


class HardwareInfo:
    @staticmethod
    def get_info() -> dict[str, Any]:
        """
        Gather system hardware specifications including CPU details and total RAM.
        """
        model_name, technical_name, core_count = CPUInfo.get_info()
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


class IntelMicrocode:
    @staticmethod
    def get_info() -> str:
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
                    result = f"13th Gen detected. Microcode revision: {microcode}"
                elif re.search(r"i\d-14", cpu_name, re.IGNORECASE):
                    result = f"14th Gen detected. Microcode revision: {microcode}"
            else:
                if re.search(r"i\d-13", cpu_name, re.IGNORECASE):
                    result = "13th Gen detected. Microcode revision: Unavailable"
                elif re.search(r"i\d-14", cpu_name, re.IGNORECASE):
                    result = "14th Gen detected. Microcode revision: Unavailable"
        except Exception as e:
            logger.exception("Error retrieving microcode information")
            result = f"Unable to retrieve microcode information: {e}"
        return result


@dataclass
class GPUInfo:
    Identifier: str
    Name: str
    Vendor: str
    ExtendedInfo: str


class GPUCollector:
    @staticmethod
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
        merged_output = Utility.run_command(cmd)
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

    @staticmethod
    def get_nvidia() -> list[GPUInfo]:
        output = Utility.run_command(["nvidia-smi", "-L"])
        if not output:
            logger.error("nvidia-smi did not return valid output.")
            return []
        nvidia_gpus = [
            {"Index": m.group(1), "Name": m.group(2)}
            for line in output.splitlines()
            if (m := re.match(r"GPU (\d+): (.*) \(UUID:", line))
        ]
        gpus = []
        for gpu in nvidia_gpus:
            extended = GPUCollector.query_nvidia_gpu_extended_info(
                gpu["Index"]
            )
            gpus.append(
                GPUInfo(
                    Identifier=f"NVIDIA GPU (Index {gpu['Index']})",
                    Name=gpu["Name"],
                    Vendor="NVIDIA",
                    ExtendedInfo=extended,
                )
            )
        return gpus

    @staticmethod
    def get_lshw() -> GPUInfo:
        """
        Parse GPU information from lshw JSON output using filtering to display class only.
        """
        try:
            # Run lshw with JSON output
            result = Utility.subprocess_run(
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
                    product_str = device.get("product", "")
                    gpu["Vendor"] = GPUCollector.determine_vendor(
                        vendor_str, product_str
                    )

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
                    gpu["ExtendedInfo"] = GPUCollector.build_extended_info(
                        extended
                    )

                    break  # Take first display device

            return GPUInfo(
                Identifier=gpu["Identifier"],
                Name=gpu["Name"],
                Vendor=gpu["Vendor"],
                ExtendedInfo=gpu["ExtendedInfo"],
            )

        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing lshw JSON output: {e}")
            return GPUInfo(
                Identifier="Error",
                Name="Error parsing GPU info",
                ExtendedInfo=str(e),
                Vendor="Unknown",
            )

    @staticmethod
    def get_lspci() -> list[GPUInfo]:
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

        gpus: list[GPUInfo] = []
        try:
            output = Utility.run_command(["lspci", "-vmm"])
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
            device_str = device_match.group(1).strip()

            # Use both vendor string and device string for detection
            vendor = GPUCollector.determine_vendor(vendor_str, device_str)

            # Skip NVIDIA GPUs since they are handled elsewhere
            if vendor == "NVIDIA":
                continue

            slot = slot_match.group(1).strip()
            gpus.append(
                GPUInfo(
                    Identifier=f"PCI Slot {slot}",
                    Name=device_str,
                    Vendor=vendor,
                    ExtendedInfo=GPUCollector.build_extended_info(
                        [f"Vendor: {vendor_str}", f"PCI Slot: {slot}"]
                    ),
                )
            )
        return gpus

    @staticmethod
    def get_generic() -> list[GPUInfo]:
        """
        Retrieve driver-specific GPU info for Intel/AMD on Windows via PowerShell.
        Returns a list of GPU info instances.
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
            result = Utility.subprocess_run(
                ["powershell", "-Command", ps_command]
            )
            output = result.stdout.strip()
            gpu_list = []
            if output:
                parsed = json.loads(output)
                gpu_list = [parsed] if isinstance(parsed, dict) else parsed
                for gpu in gpu_list:
                    identifier = gpu.get("Name", "Unknown GPU")
                    vendor = gpu.get("AdapterCompatibility", "Unknown")
                    gpu["Identifier"] = identifier
                    gpu["Vendor"] = vendor
                return [
                    GPUInfo(
                        Identifier=gpu.get("Identifier", "Unknown"),
                        Name=gpu.get("Name", "Unknown"),
                        Vendor=gpu.get("Vendor", "Unknown"),
                        ExtendedInfo=f"Driver Version: {gpu.get('DriverVersion', 'Unavailable')}",
                    )
                    for gpu in gpu_list
                ]
        except Exception:
            logger.exception("Error retrieving Intel/AMD GPU info")
        return []

    @staticmethod
    def get_macos() -> list[GPUInfo]:
        """
        Retrieve GPU information on macOS using system_profiler.
        Returns a list of instances containing GPU details.
        """
        gpus: list[GPUInfo] = []
        try:
            output = Utility.run_command(
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
                    extended_info = GPUCollector.build_extended_info(
                        extended_info_parts
                    )
                    gpus.append(
                        GPUInfo(
                            Identifier=identifier,
                            Name=gpu_name,
                            Vendor=vendor,
                            ExtendedInfo=extended_info,
                        )
                    )
        except Exception:
            logger.exception("Error retrieving GPU info on macOS")
        return gpus

    @staticmethod
    def determine_vendor(vendor_str: str, device_name: str = "") -> str:
        """
        Determine the GPU vendor from vendor string and optionally device name.
        Takes both the vendor string and device name into account for more accurate detection.
        """
        lower_vendor = vendor_str.lower() if vendor_str else ""
        lower_device = device_name.lower() if device_name else ""

        # Check for NVIDIA keywords in either string
        if "nvidia" in lower_vendor or any(
            x in lower_device
            for x in {"geforce", "rtx", "gtx", "quadro", "tesla", "titan"}
        ):
            return "NVIDIA"
        # Check for AMD keywords
        elif any(
            x in lower_vendor
            for x in {
                "amd",
                "ati",
                "advanced micro",
            }
        ) or any(x in lower_device for x in {"radeon", "firepro"}):
            return "AMD"
        # Check for Intel
        elif "intel" in lower_vendor or "intel" in lower_device:
            return "Intel"

        return "Unknown"

    @staticmethod
    def build_extended_info(info_items: list[str]) -> str:
        return (
            "\n".join("    " + item for item in info_items)
            if info_items
            else "No extended info available"
        )

    @staticmethod
    def get_all() -> list[GPUInfo]:
        """
        Gather GPU information using a multi-stage approach.
        First Mac, then NVIDIA, then Intel/AMD on Windows,
        then attempt to use lspci; if that also fails,
        fall back to parsing lshw JSON output.
        """
        try:
            system = platform.system()

            # x86 MacOS (Old and Hackintoshes) are not supported. M-series macs are all thats accounted for.
            if system == "Darwin":
                return Utility.safe_call(GPUCollector.get_macos, [])

            # Obviously NVIDIA GPUs are the most common, so we check for them first after short-circuiting for Mac.
            # Approach works for both Windows and Linux.
            nvidia_gpus = Utility.safe_call(GPUCollector.get_nvidia, [])
            if nvidia_gpus:
                return nvidia_gpus

            # Next most common case is Intel/AMD GPUs on Windows, check via powershell
            if system == "Windows":
                return Utility.safe_call(GPUCollector.get_generic, [])

            # If not nvidia and is Linux, try lspci first, then fall back to lshw.
            if system == "Linux":
                lspci_gpus = Utility.safe_call(GPUCollector.get_lspci, [])
                if lspci_gpus:
                    return lspci_gpus

                lshw_gpu = Utility.safe_call(GPUCollector.get_lshw, None)
                if (
                    lshw_gpu
                    and lshw_gpu.Name
                    and lshw_gpu.Name != "Unknown"
                ):
                    return [lshw_gpu]

            return []
        except Exception as e:
            logger.exception(f"Error in get_gpu_info: {e}")
            return [
                GPUInfo(
                    Identifier="Unknown",
                    Name="Unknown GPU",
                    Vendor="Unknown",
                    ExtendedInfo=GPUCollector.build_extended_info([]),
                )
            ]


class SoftwareInfo:
    @staticmethod
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
        pip_out = Utility.run_command(cmd)
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

    @staticmethod
    def get_git_info() -> str:
        """
        Retrieve Git information including repository details (user and repo name),
        the current branch, commit hash, and any modified files compared to the upstream branch.
        """
        # Get remote repository URL info for fork detection.
        remote_url = Utility.run_command(
            ["git", "config", "--get", "remote.origin.url"]
        )
        if remote_url:
            import re

            match = re.search(
                r"[:/](?P<user>[^/]+)/(?P<repo>[^/.]+)(\.git)?$",
                remote_url,
            )
            repo_info = (
                f"Repo: {match.group('user')}/{match.group('repo')}"
                if match
                else f"Remote URL: {remote_url}"
            )
        else:
            repo_info = "No remote repository URL available."

        branch = Utility.run_command(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        )
        if branch is None:
            return "Not a Git repository or git not installed."

        commit = (
            Utility.run_command(["git", "rev-parse", "HEAD"])
            or "Unavailable"
        )
        # Insert repo info at the top, followed by branch and commit.
        git_info = f"{repo_info}\nBranch: {branch}\nCommit: {commit}"

        # Check for untracked files
        untracked_files = Utility.run_command(
            ["git", "ls-files", "--others", "--exclude-standard"]
        )
        if untracked_files and untracked_files.strip():
            untracked_list = "\n".join(
                f"  {line}" for line in untracked_files.splitlines()
            )
            git_info += f"\nUntracked Files:\n{untracked_list}"
        else:
            git_info += "\nNo untracked files."

        # Check for modified files relative to upstream
        upstream = Utility.run_command(
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
                Utility.run_command(["git", "diff", "--name-only", "@{u}"])
                or ""
            )
            if diff_files.strip():
                modified_files = "\n".join(
                    f"  {line}" for line in diff_files.splitlines()
                )
                git_info += f"\nModified Files (differs from {upstream}):\n{modified_files}"
            else:
                git_info += f"\nNo modifications relative to upstream ({upstream})."
        else:
            git_info += (
                "\nNo upstream branch tracking information available."
            )

        return git_info


class NetworkInfo:
    @staticmethod
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
            result = Utility.subprocess_run(cmd)
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

    @staticmethod
    def test_connectivity() -> dict[str, tuple[str, str]]:
        """Test connectivity to key services."""
        urls = {
            "PyPI": "https://pypi.org/",
            "HuggingFace": "https://huggingface.co",
            "Google": "https://www.google.com",
        }
        return {
            name: (url, NetworkInfo.test_url(url))
            for name, url in urls.items()
        }


class ReportBuilder:
    @staticmethod
    def build_report() -> list[str]:
        """
        Collect system information and build the debug report.
        """
        os_info = OSInfo.get_info()
        hardware_info = HardwareInfo.get_info()
        gpu_info = GPUCollector.get_all()
        python_info = SoftwareInfo.get_python_info()
        git_info = SoftwareInfo.get_git_info()
        network_status = NetworkInfo.test_connectivity()
        intel_microcode = IntelMicrocode.get_info()

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
                    f"{gpu.Identifier}: {gpu.Name} [{gpu.Vendor}]"
                )
                if gpu.ExtendedInfo:
                    report.append(gpu.ExtendedInfo)
        else:
            report.append("No GPUs detected.")
        report.extend(
            [
                "",
                "=== Python Environment ===",
                f"Global Python Version: {python_info.get('PythonVersion', 'Unavailable')}",
                f"Python Executable Path: {Utility.anonymize_path(python_info.get('PythonPath'))}",
                f"PyTorch Info: {python_info.get('PyTorchInfo', 'Unavailable')}",
                "pip freeze output:",
                python_info.get("PipFreeze", "Unavailable"),
                "",
                "=== Git Information ===",
                git_info,
                "",
                "=== Network Connectivity ===",
                *[
                    f"{name} ({url}): {status}"
                    for name, (url, status) in network_status.items()
                ],
                "",
                "=== Intel Microcode Information ===",
                intel_microcode,
                "",
            ]
        )
        return report

    @staticmethod
    def write_report(report: list[str], output_file: Path) -> None:
        """
        Write the report to the given output file with final anonymization.
        """
        try:
            anonymized_report = "\n".join(
                Utility.anonymize_path(line) for line in report
            )
            output_file.write_text(anonymized_report, encoding="utf-8")
            logger.info(f"Report assembled and saved to {output_file}")
        except Exception:
            logger.exception("Failed to write report to file")


def main() -> None:
    """Main function to collect info, build the debug report, and write it to a file."""

    current_dir = Path.cwd()
    logger.info(f"Current directory: {current_dir}")
    if current_dir.name != "OneTrainer":
        logger.warning(
            f"Expected to run from the OneTrainer folder. Current folder: {current_dir}"
        )

    try:
        report = ReportBuilder.build_report()
        output_file = Path("debug_report.log")
        ReportBuilder.write_report(report, output_file)
        print(
            f"Debug report created successfully: {output_file.absolute()}"
        )
    except Exception as e:
        logger.error(f"Failed to generate debug report: {e}")
        raise


if __name__ == "__main__":
    main()
