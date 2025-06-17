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
    // Define priority order between CPU Features (Lowest number == Highest Priority)
    fn priority(&self) -> usize {
        match self.name {
            "avx512f" => 0,
            "avx2" => 1,
            "sse4_1" => 2,
            _ => usize::MAX, // lowest priority by default
        }
    }

    // Groups all supported CPU features that use optimizations in this crate
    // used in stable build only
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
                rustc_flag: "+avx2,+avx",
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

    // Groups all supported CPU features that use optimizations in this crate
    // used in nightly build only
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
                rustc_flag: "+avx2,+avx",
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

// Feature detection trait to make implementations more modular
trait CpuFeatureDetector {
    fn detect_features(&self, features: &mut [CpuFeature]);
    fn is_applicable(&self) -> bool;
}

// Linux CPU feature detector
struct LinuxDetector;
impl CpuFeatureDetector for LinuxDetector {
    fn detect_features(&self, features: &mut [CpuFeature]) {
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            let contents = cpuinfo.to_lowercase();
            for feature in features.iter_mut() {
                feature.detected = contents.contains(feature.name);
            }
        }
    }

    fn is_applicable(&self) -> bool {
        cfg!(target_os = "linux")
    }
}

// macOS CPU feature detector
struct MacOSDetector;
impl CpuFeatureDetector for MacOSDetector {
    fn detect_features(&self, features: &mut [CpuFeature]) {
        let output = Command::new("sysctl").args(["-a"]).output();

        if let Ok(output) = output {
            let contents = String::from_utf8_lossy(&output.stdout).to_lowercase();

            for feature in features.iter_mut() {
                match feature.name {
                    "avx512f" => feature.detected = contents.contains("hw.optional.avx512f: 1"),
                    "avx2" => feature.detected = contents.contains("hw.optional.avx2_0: 1"),
                    "sse4_1" => feature.detected = contents.contains("hw.optional.sse4_1: 1"),
                    "neon" => feature.detected = contents.contains("hw.optional.neon: 1"),
                    _ => {}
                }
            }
        }
    }

    fn is_applicable(&self) -> bool {
        cfg!(target_os = "macos")
    }
}

// No windows detector for now
// TODO: Develop a Windows detector (Access to a windows machine needed)

// Factory that creates the appropriate detector for the current OS
struct PlatformDetector;
impl PlatformDetector {
    fn cpu_features_detectors() -> Vec<Box<dyn CpuFeatureDetector>> {
        vec![Box::new(LinuxDetector), Box::new(MacOSDetector)]
    }

    fn compiler_channel() -> String {
        let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
        let output = Command::new(rustc)
            .args(["--version", "--verbose"])
            .output()
            .expect("Failed to execute rustc");

        let version_info = String::from_utf8_lossy(&output.stdout);

        if version_info.contains("nightly") {
            "nightly".to_string()
        } else {
            "stable".to_string()
        }
    }

    fn detect_cpu_features(features: &mut [CpuFeature]) {
        // Get detectors for all supported platforms
        let detectors = Self::cpu_features_detectors();

        // Find the applicable detector and use it
        for detector in detectors {
            if detector.is_applicable() {
                detector.detect_features(features);
                break;
            }
        }
    }

    fn apply(features: &mut [CpuFeature]) {
        // Sort features by priority (highest first)
        features.sort();

        // Find and use the highest detected feature (if any)
        // if no feature is detected, use fallback implementation
        let cfg_flag = features
            .iter()
            .find(|cpu_feature| cpu_feature.detected)
            .map(|cpu_feature| {
                println!("cargo:rustc-flag=-C");
                println!("cargo:rustc-flag=target-feature={}", cpu_feature.rustc_flag);
                cpu_feature.cfg_flag
            })
            .unwrap_or_else(|| "fallback");

        println!("applying: {cfg_flag}");

        println!("cargo:rustc-cfg={cfg_flag}");

        println!("cargo::rustc-check-cfg=cfg(avx512)");
        println!("cargo::rustc-check-cfg=cfg(avx2)");
        println!("cargo::rustc-check-cfg=cfg(sse)");
        println!("cargo::rustc-check-cfg=cfg(neon)");
        println!("cargo::rustc-check-cfg=cfg(fallback)");
    }
}

fn main() {
    // Detect rustc channel (stable, beta, nightly)
    let rustc_channel = PlatformDetector::compiler_channel();

    // Create a flag for modules that can be used in nighlty build only
    // Some features like avx512 are available only with nighlty build
    println!("cargo:rustc-cfg=rustc_channel=\"{rustc_channel}\"");

    // Disable flag warnings for build
    println!("cargo::rustc-check-cfg=cfg(rustc_channel, values(\"nightly\", \"stable\"))");

    let nightly_build = rustc_channel == "nightly";

    // Define the CPU features we're interested in (channel dependant)
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
        PlatformDetector::detect_cpu_features(&mut features);
    }

    // Pass RUSTFLAGS for enabling target features
    PlatformDetector::apply(&mut features);
}
