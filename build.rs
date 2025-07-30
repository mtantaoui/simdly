//! Build script for automatic CPU feature detection and optimization.
//!
//! This build script automatically detects available CPU features on the target system
//! and enables appropriate SIMD optimizations during compilation. It supports multiple
//! platforms and provides fallback implementations when advanced features are unavailable.
//!
//! # Supported Features
//!
//! - **AVX2**: Intel Advanced Vector Extensions 2 (256-bit vectors)
//! - **NEON**: ARM Advanced SIMD (128-bit vectors)  
//! - **SSE4.1**: Streaming SIMD Extensions 4.1 (128-bit vectors)
//! - **AVX-512**: Intel Advanced Vector Extensions 512 (512-bit vectors, nightly only)
//!
//! # Platform Support
//!
//! - **Linux**: Reads `/proc/cpuinfo` for feature detection
//! - **macOS**: Uses `sysctl` to query hardware capabilities
//! - **Windows**: Uses PowerShell and WMI for CPU feature detection
//!
//! # Cross-Compilation
//!
//! When cross-compiling, the build script skips CPU detection and falls back to
//! generic implementations to ensure compatibility with the target architecture.

use std::cmp::Ordering;
use std::env;
use std::process::Command;

// CPU features we want to detect
#[derive(PartialEq, Eq, Debug)]
struct CpuFeature {
    name: &'static str,
    rustc_flag: &'static str,
    cfg_flag: &'static str,
    detected: bool,
    nightly_only: bool,
}

impl CpuFeature {
    /// Define priority order between CPU Features (Lowest number == Highest Priority)
    /// 
    /// Returns the priority value for feature selection, where lower numbers indicate
    /// higher priority. This ensures the most advanced compatible instruction set is used.
    #[inline(always)]
    fn priority(&self) -> usize {
        match self.name {
            "avx512f" => 0,
            "avx2" => 1,
            "sse4_1" => 2,
            _ => usize::MAX, // lowest priority by default
        }
    }

    /// Groups all supported CPU features that use optimizations in this crate.
    /// 
    /// Used in stable build only. Returns a vector of CPU features that can be
    /// safely used with the stable Rust compiler.
    #[inline(always)]
    fn features() -> Vec<CpuFeature> {
        vec![
            CpuFeature {
                name: "sse4_1",
                rustc_flag: "+sse4.1",
                cfg_flag: "sse",
                detected: false,
                nightly_only: false,
            },
            CpuFeature {
                name: "avx2",
                rustc_flag: "+avx2,+avx,+fma",
                cfg_flag: "avx2",
                detected: false,
                nightly_only: false,
            },
            CpuFeature {
                name: "neon",
                rustc_flag: "+neon",
                cfg_flag: "neon",
                detected: false,
                nightly_only: false,
            },
        ]
    }

    /// Groups all supported CPU features that use optimizations in this crate.
    /// 
    /// Used in nightly build only. Includes experimental features that require
    /// nightly Rust compiler features.
    #[inline(always)]
    fn nightly_features() -> Vec<CpuFeature> {
        vec![
            CpuFeature {
                name: "sse4_1",
                rustc_flag: "+sse4.1",
                cfg_flag: "sse",
                detected: false,
                nightly_only: false,
            },
            CpuFeature {
                name: "avx512f",
                rustc_flag: "+avx512f",
                cfg_flag: "avx512",
                detected: false,
                nightly_only: true,
            },
            CpuFeature {
                name: "avx2",
                rustc_flag: "+avx2,+avx,+fma",
                cfg_flag: "avx2",
                detected: false,
                nightly_only: false,
            },
            CpuFeature {
                name: "neon",
                rustc_flag: "+neon",
                cfg_flag: "neon",
                detected: false,
                nightly_only: false,
            },
        ]
    }
}

impl Ord for CpuFeature {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority().cmp(&other.priority())
    }
}

impl PartialOrd for CpuFeature {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Result type for feature detection
#[derive(Debug)]
enum DetectionResult {
    Success,
    PartialFailure(String), // Some features detected, but with warnings
    Failure(String),        // Complete failure with error message
}

// Feature detection trait to make implementations more modular
trait CpuFeatureDetector {
    fn detect_features(&self, features: &mut [CpuFeature]) -> DetectionResult;
    fn is_applicable(&self) -> bool;
    fn name(&self) -> &'static str;
}

// Linux CPU feature detector
struct LinuxDetector;
impl CpuFeatureDetector for LinuxDetector {
    #[inline(always)]
    fn detect_features(&self, features: &mut [CpuFeature]) -> DetectionResult {
        match std::fs::read_to_string("/proc/cpuinfo") {
            Ok(cpuinfo) => {
                let contents = cpuinfo.to_lowercase();
                for feature in features.iter_mut() {
                    feature.detected = contents.contains(feature.name);
                }
                DetectionResult::Success
            }
            Err(e) => DetectionResult::Failure(format!("Failed to read /proc/cpuinfo: {e}")),
        }
    }

    #[inline(always)]
    fn is_applicable(&self) -> bool {
        cfg!(target_os = "linux")
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        "Linux"
    }
}

// macOS CPU feature detector
struct MacOSDetector;
impl CpuFeatureDetector for MacOSDetector {
    #[inline(always)]
    fn detect_features(&self, features: &mut [CpuFeature]) -> DetectionResult {
        let output = Command::new("sysctl").args(["-a"]).output();

        match output {
            Ok(output) => {
                if !output.status.success() {
                    return DetectionResult::Failure(format!(
                        "sysctl command failed with exit code: {}",
                        output.status
                    ));
                }

                let contents = String::from_utf8_lossy(&output.stdout).to_lowercase();
                let mut detected_any = false;

                for feature in features.iter_mut() {
                    match feature.name {
                        "avx512f" => {
                            feature.detected = contents.contains("hw.optional.avx512f: 1");
                        }
                        "avx2" => {
                            feature.detected = contents.contains("hw.optional.avx2_0: 1");
                        }
                        "sse4_1" => {
                            feature.detected = contents.contains("hw.optional.sse4_1: 1");
                        }
                        "neon" => {
                            feature.detected = contents.contains("hw.optional.neon: 1");
                        }
                        _ => {}
                    }
                    if feature.detected {
                        detected_any = true;
                    }
                }

                if detected_any {
                    DetectionResult::Success
                } else {
                    DetectionResult::PartialFailure(
                        "No CPU features detected in sysctl output".to_string(),
                    )
                }
            }
            Err(e) => DetectionResult::Failure(format!("Failed to execute sysctl: {e}")),
        }
    }

    #[inline(always)]
    fn is_applicable(&self) -> bool {
        cfg!(target_os = "macos")
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        "macOS"
    }
}

// Windows CPU feature detector
struct WindowsDetector;
impl CpuFeatureDetector for WindowsDetector {
    #[inline(always)]
    fn detect_features(&self, features: &mut [CpuFeature]) -> DetectionResult {
        // Try PowerShell method first (more reliable)
        if let Ok(result) = self.detect_with_powershell(features) {
            return result;
        }

        // Fallback to wmic method
        self.detect_with_wmic(features)
    }

    #[inline(always)]
    fn is_applicable(&self) -> bool {
        cfg!(target_os = "windows")
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        "Windows"
    }
}

impl WindowsDetector {
    /// Attempts to detect CPU features using PowerShell and WMI.
    /// 
    /// This method provides more reliable CPU feature detection on Windows
    /// by querying detailed processor information through WMI.
    #[inline(always)]
    fn detect_with_powershell(&self, features: &mut [CpuFeature]) -> Result<DetectionResult, ()> {
        let powershell_script = r#"
            try {
                $cpu = Get-WmiObject -Class Win32_Processor | Select-Object -First 1
                $features = @{}
                
                # Get processor name and features
                $name = $cpu.Name.ToLower()
                
                # Basic feature detection based on processor name and generation
                if ($name -match "intel") {
                    # Intel processors
                    if ($name -match "core.*i[3-9]" -or $name -match "xeon") {
                        $features["sse4_1"] = $true
                        if ($name -match "haswell|broadwell|skylake|kaby.*lake|coffee.*lake|ice.*lake|tiger.*lake|alder.*lake|raptor.*lake" -or 
                            $name -match "i[5-9].*[2-9][0-9][0-9][0-9]" -or $name -match "xeon.*e[3-7]") {
                            $features["avx2"] = $true
                        }
                        if ($name -match "skylake.*x|cascade.*lake|cooper.*lake|ice.*lake.*sp|sapphire.*rapids" -or 
                            $name -match "xeon.*platinum|xeon.*gold") {
                            $features["avx512f"] = $true
                        }
                    }
                } elseif ($name -match "amd") {
                    # AMD processors
                    if ($name -match "ryzen|epyc|threadripper") {
                        $features["sse4_1"] = $true
                        $features["avx2"] = $true
                    }
                } elseif ($name -match "arm|qualcomm|snapdragon") {
                    # ARM processors
                    $features["neon"] = $true
                }
                
                # Output results
                foreach ($feature in $features.Keys) {
                    if ($features[$feature]) {
                        Write-Output "$feature=true"
                    }
                }
                
                Write-Output "detection=success"
            } catch {
                Write-Output "detection=error:$($_.Exception.Message)"
            }
        "#;

        let output = Command::new("powershell")
            .args(["-NoProfile", "-Command", powershell_script])
            .output()
            .map_err(|_| ())?;

        if !output.status.success() {
            return Err(());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut detected_any = false;

        for line in stdout.lines() {
            if line.starts_with("detection=error:") {
                return Ok(DetectionResult::Failure(
                    line.strip_prefix("detection=error:")
                        .unwrap_or("Unknown PowerShell error")
                        .to_string(),
                ));
            } else if line == "detection=success" {
                continue;
            } else if let Some((feature_name, value)) = line.split_once('=') {
                if value == "true" {
                    for feature in features.iter_mut() {
                        if feature.name == feature_name {
                            feature.detected = true;
                            detected_any = true;
                            break;
                        }
                    }
                }
            }
        }

        Ok(if detected_any {
            DetectionResult::Success
        } else {
            DetectionResult::PartialFailure("No CPU features detected via PowerShell".to_string())
        })
    }

    /// Fallback method using wmic command for CPU feature detection.
    /// 
    /// Uses basic heuristics based on processor name patterns to determine
    /// supported instruction sets when PowerShell detection fails.
    #[inline(always)]
    fn detect_with_wmic(&self, features: &mut [CpuFeature]) -> DetectionResult {
        let output = Command::new("wmic")
            .args(["cpu", "get", "name", "/format:list"])
            .output();

        match output {
            Ok(output) => {
                if !output.status.success() {
                    return DetectionResult::Failure(format!(
                        "wmic command failed with exit code: {}",
                        output.status
                    ));
                }

                let contents = String::from_utf8_lossy(&output.stdout).to_lowercase();
                let mut detected_any = false;

                // Basic heuristic-based detection for common processors
                for feature in features.iter_mut() {
                    match feature.name {
                        "sse4_1" => {
                            feature.detected = contents.contains("intel core i")
                                || contents.contains("intel xeon")
                                || contents.contains("amd ryzen")
                                || contents.contains("amd epyc");
                        }
                        "avx2" => {
                            feature.detected = (contents.contains("intel core i")
                                && (contents.contains("4th gen")
                                    || contents.contains("haswell")
                                    || contents.contains("5th gen")
                                    || contents.contains("broadwell")
                                    || contents.contains("6th gen")
                                    || contents.contains("skylake")
                                    || contents.contains("7th gen")
                                    || contents.contains("kaby")
                                    || contents.contains("8th gen")
                                    || contents.contains("coffee")
                                    || contents.contains("9th gen")
                                    || contents.contains("10th gen")
                                    || contents.contains("11th gen")
                                    || contents.contains("12th gen")
                                    || contents.contains("13th gen")))
                                || contents.contains("amd ryzen")
                                || contents.contains("amd epyc");
                        }
                        "avx512f" => {
                            feature.detected = contents.contains("intel xeon")
                                && (contents.contains("skylake")
                                    || contents.contains("cascade")
                                    || contents.contains("cooper")
                                    || contents.contains("ice lake"));
                        }
                        "neon" => {
                            feature.detected = contents.contains("arm")
                                || contents.contains("qualcomm")
                                || contents.contains("snapdragon");
                        }
                        _ => {}
                    }
                    if feature.detected {
                        detected_any = true;
                    }
                }

                if detected_any {
                    DetectionResult::Success
                } else {
                    DetectionResult::PartialFailure("No CPU features detected via wmic".to_string())
                }
            }
            Err(e) => DetectionResult::Failure(format!("Failed to execute wmic: {e}")),
        }
    }
}

// Factory that creates the appropriate detector for the current OS
struct PlatformDetector;
impl PlatformDetector {
    /// Creates a vector of all available CPU feature detectors.
    /// 
    /// Returns detectors for all supported platforms. The appropriate detector
    /// will be selected based on the current operating system.
    #[inline(always)]
    fn cpu_features_detectors() -> Vec<Box<dyn CpuFeatureDetector>> {
        vec![
            Box::new(LinuxDetector),
            Box::new(MacOSDetector),
            Box::new(WindowsDetector),
        ]
    }

    /// Detects the current Rust compiler channel (stable, beta, nightly).
    /// 
    /// This information is used to determine which CPU features can be safely
    /// enabled, as some features require nightly compiler support.
    #[inline(always)]
    fn compiler_channel() -> Result<String, String> {
        let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());

        let output = Command::new(&rustc)
            .args(["--version", "--verbose"])
            .output()
            .map_err(|e| format!("Failed to execute rustc ({rustc}): {e}"))?;

        if !output.status.success() {
            return Err(format!(
                "rustc command failed with exit code: {}",
                output.status
            ));
        }

        let version_info = String::from_utf8_lossy(&output.stdout);

        Ok(if version_info.contains("nightly") {
            "nightly".to_string()
        } else {
            "stable".to_string()
        })
    }

    /// Performs CPU feature detection using the appropriate platform detector.
    /// 
    /// Iterates through available detectors and uses the first applicable one
    /// for the current platform to detect supported CPU features.
    #[inline(always)]
    fn detect_cpu_features(features: &mut [CpuFeature]) -> DetectionResult {
        let detectors = Self::cpu_features_detectors();
        let mut last_error = String::new();

        // Find the applicable detector and use it
        for detector in detectors {
            if detector.is_applicable() {
                println!(
                    "cargo:warning=Using {} CPU feature detector",
                    detector.name()
                );

                match detector.detect_features(features) {
                    DetectionResult::Success => {
                        println!("cargo:warning=CPU feature detection successful");
                        return DetectionResult::Success;
                    }
                    DetectionResult::PartialFailure(msg) => {
                        println!("cargo:warning=CPU feature detection partial failure: {msg}");
                        return DetectionResult::PartialFailure(msg);
                    }
                    DetectionResult::Failure(msg) => {
                        last_error = msg;
                        println!("cargo:warning=CPU feature detection failed: {last_error}");
                        continue;
                    }
                }
            }
        }

        // If we reach here, no detector was applicable or all failed
        if last_error.is_empty() {
            DetectionResult::Failure(
                "No applicable CPU feature detector found for this platform".to_string(),
            )
        } else {
            DetectionResult::Failure(last_error)
        }
    }

    /// Applies the detected CPU features to the build configuration.
    /// 
    /// Sorts features by priority, selects the best available feature,
    /// and configures the appropriate Rust compiler flags and cfg attributes.
    #[inline(always)]
    fn apply(features: &mut [CpuFeature]) {
        // Sort features by priority (highest first)
        features.sort();

        // Find and use the highest detected feature (if any)
        let cfg_flag = features
            .iter()
            .find(|cpu_feature| cpu_feature.detected)
            .map(|cpu_feature| {
                println!("cargo:rustc-flag=-C");
                println!("cargo:rustc-flag=target-feature={}", cpu_feature.rustc_flag);
                println!(
                    "cargo:warning=Enabled CPU feature optimization: {} ({})",
                    cpu_feature.name, cpu_feature.rustc_flag
                );
                cpu_feature.cfg_flag
            })
            .unwrap_or_else(|| {
                println!("cargo:warning=No CPU features detected, using fallback implementation");
                "fallback"
            });

        println!("cargo:rustc-cfg={cfg_flag}");

        // Configure check-cfg for all possible configurations
        println!("cargo::rustc-check-cfg=cfg(avx512)");
        println!("cargo::rustc-check-cfg=cfg(avx2)");
        println!("cargo::rustc-check-cfg=cfg(sse)");
        println!("cargo::rustc-check-cfg=cfg(neon)");
        println!("cargo::rustc-check-cfg=cfg(fallback)");
    }
}

/// Main build script entry point.
///
/// Orchestrates the entire CPU feature detection and optimization process:
/// 1. Detects the Rust compiler channel (stable/nightly)
/// 2. Determines if we're cross-compiling
/// 3. Runs CPU feature detection for native builds
/// 4. Applies the appropriate compiler optimizations
#[inline(always)]
fn main() {
    // Detect rustc channel (stable, beta, nightly)
    let rustc_channel = match PlatformDetector::compiler_channel() {
        Ok(channel) => channel,
        Err(e) => {
            println!("cargo:warning=Failed to detect rustc channel: {e}");
            println!("cargo:warning=Defaulting to stable channel");
            "stable".to_string()
        }
    };

    // Create a flag for modules that can be used in nightly build only
    println!("cargo:rustc-cfg=rustc_channel=\"{rustc_channel}\"");
    println!("cargo::rustc-check-cfg=cfg(rustc_channel, values(\"nightly\", \"stable\"))");

    let nightly_build = rustc_channel == "nightly";

    // Define the CPU features we're interested in (channel dependent)
    let mut features = if nightly_build {
        CpuFeature::nightly_features()
    } else {
        CpuFeature::features()
    };

    // Determine if we're cross-compiling
    let host = env::var("HOST").unwrap_or_default();
    let target = env::var("TARGET").unwrap_or_default();
    let is_native_build = host == target;

    // Only run CPU detection for native builds
    if is_native_build {
        match PlatformDetector::detect_cpu_features(&mut features) {
            DetectionResult::Success => {
                // Features detected successfully
            }
            DetectionResult::PartialFailure(msg) => {
                println!("cargo:warning=CPU feature detection partially failed: {msg}",);
                println!("cargo:warning=Some optimizations may not be available");
            }
            DetectionResult::Failure(msg) => {
                println!("cargo:warning=CPU feature detection failed: {msg}");
                println!("cargo:warning=Falling back to default implementation");

                // Reset all features to undetected for safety
                for feature in features.iter_mut() {
                    feature.detected = false;
                }
            }
        }
    } else {
        println!("cargo:warning=Cross-compiling detected (host: {host}, target: {target})",);
        println!("cargo:warning=Skipping CPU feature detection, using fallback implementation");
    }

    // Apply the detected features (or fallback if none detected)
    PlatformDetector::apply(&mut features);
}
