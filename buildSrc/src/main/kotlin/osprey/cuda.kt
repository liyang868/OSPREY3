package osprey

import org.gradle.api.Project
import org.gradle.api.tasks.Exec
import org.gradle.kotlin.dsl.creating
import org.gradle.kotlin.dsl.getValue
import java.io.File


fun Project.makeCudaTasks() {

	val compileCuda_forcefield by tasks.creating(Exec::class) {
		nvcc(this, "forcefield")
	}

	val compileCuda_ccd by tasks.creating(Exec::class) {
		nvcc(this, "ccd")
	}

	val compileCuda_residueForcefield by tasks.creating(Exec::class) {
		nvcc(this, "residueForcefield")
	}

	val compileCuda_residueCcd by tasks.creating(Exec::class) {
		nvcc(this, "residueCcd", maxRegisters=64)
	}


	val compileCuda_dp by tasks.creating(Exec::class) {
		nvcc(this, "dp")
	}

	val compileCuda_sampling by tasks.creating(Exec::class) {
		nvcc(this, "sampling")
	}

	@Suppress("UNUSED_VARIABLE")
	val compileCuda by tasks.creating {
		description = "Compile cuda kernels"
		dependsOn(
			compileCuda_forcefield,
			compileCuda_ccd,
			compileCuda_residueForcefield,
			compileCuda_residueCcd,
			compileCuda_dp,
			compileCuda_sampling
		)
	}
}

fun Project.nvcc(exec: Exec, kernelName: String, maxRegisters: Int? = null, profile: Boolean = false) {

	val nvcc = listOfNotNull(
		System.getenv("NVCC"),
		System.getenv("CUDA_HOME")?.let { "$it/bin/nvcc" },
		"nvcc"
	).firstOrNull { it == "nvcc" || File(it).canExecute() } ?: "nvcc"

	val args = mutableListOf(nvcc)

	if (profile) {
		// if profiling, compile for one arch with profiling/debug info
		// NOTE: change this to your GPU's arch
		args.addAll(listOf("-cubin", "-gencode=arch=compute_86,code=sm_86", "-lineinfo", "--ptxas-options=-v"))
	} else {
		// Otherwise, compile for V100/Titan V and Ampere A5000.  Hopper's
		// compute_90 target is only accepted by CUDA >= 11.8; older nvcc
		// installations are still common on the cluster and fail the entire
		// build when handed an unknown architecture.  Allow an explicit
		// OSPREY_CUDA_ARCHS override, and add 90 automatically when supported.
		val requestedArchs = System.getenv("OSPREY_CUDA_ARCHS")
		val archs = if (!requestedArchs.isNullOrBlank()) {
			requestedArchs.split(',').map { it.trim() }.filter { it.isNotEmpty() }
		} else {
			val defaults = mutableListOf("70", "86")
			if (nvccSupportsCompute90(nvcc)) defaults.add("90")
			defaults
		}
		args.add("-fatbin")
		for (arch in archs) {
			args.add("-gencode=arch=compute_$arch,code=sm_$arch")
		}
		// Keep PTX for the newest target so a newer driver can JIT it when
		// requested explicitly (e.g. OSPREY_CUDA_ARCHS=90).
		if (archs.contains("90")) {
			args.add("-gencode=arch=compute_90,code=compute_90")
		}
	}

	if (maxRegisters != null) {
		args.addAll(listOf("-maxrregcount", "$maxRegisters"))
	}

	args.addAll(listOf("$kernelName.cu", "-o", "$kernelName.bin"))

	exec.workingDir = file("src/main/resources/gpuKernels/cuda")
	exec.commandLine(args)
}

private fun nvccSupportsCompute90(nvcc: String): Boolean {
	return try {
		val process = ProcessBuilder(nvcc, "--version")
			.redirectErrorStream(true)
			.start()
		val output = process.inputStream.bufferedReader().readText()
		process.waitFor()
		val match = Regex("release\\s+(\\d+)\\.(\\d+)", RegexOption.IGNORE_CASE)
			.find(output)
		val major = match?.groupValues?.getOrNull(1)?.toIntOrNull() ?: return false
		val minor = match.groupValues.getOrNull(2)?.toIntOrNull() ?: return false
		major > 11 || (major == 11 && minor >= 8)
	} catch (_: Throwable) {
		false
	}
}
