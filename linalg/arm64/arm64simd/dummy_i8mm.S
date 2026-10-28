// Build-time capability probe for the assembler, used by build.rs
// (assembler_supports_i8mm). Assemblers predating FEAT_I8MM cannot encode `smmla`. If this
// file fails to assemble, build.rs skips the SMMLA kernel and the `tract_arm64_i8mm` cfg, and
// the runtime falls back to the SDOT or SMLAL i32 kernels. Not linked into anything.
.arch armv8.2-a+i8mm
.text
.globl tract_i8mm_probe
tract_i8mm_probe:
    smmla v0.4s, v1.16b, v2.16b
    ret
