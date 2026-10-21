package edu.duke.cs.osprey.packstar;

import edu.duke.cs.osprey.confspace.Sequence;
import edu.duke.cs.osprey.confspace.SimpleConfSpace;
import edu.duke.cs.osprey.confspace.Strand;
import edu.duke.cs.osprey.ematrix.EnergyMatrix;
import edu.duke.cs.osprey.ematrix.SimplerEnergyMatrixCalculator;
import edu.duke.cs.osprey.energy.ConfEnergyCalculator;
import edu.duke.cs.osprey.energy.EnergyCalculator;
import edu.duke.cs.osprey.energy.forcefield.ForcefieldParams;
import edu.duke.cs.osprey.kstar.KStar;
import edu.duke.cs.osprey.kstar.pfunc.PartitionFunction;
import edu.duke.cs.osprey.packstar.PackStarPartitionFunction;

import edu.duke.cs.osprey.parallelism.Parallelism;
import edu.duke.cs.osprey.restypes.ResidueTemplateLibrary;
import edu.duke.cs.osprey.structure.Molecule;
import edu.duke.cs.osprey.structure.PDBIO;

import java.io.*;
import java.util.*;

/**
 * PACK* design and partition-function runner for prepared PDB systems.
 *
 * Reads design specification from system properties:
 *   packstar.design.pdbPath       — path to prepped PDB
 *   packstar.design.proteinChains — comma-separated chain IDs for protein (e.g. "A,B")
 *   packstar.design.ligandChains  — comma-separated chain IDs for ligand (e.g. "C")
 *   packstar.design.mutable       — semicolon-separated mutable residue IDs (e.g. "A96;B85")
 *   packstar.design.flexible      — semicolon-separated flexible residue IDs (e.g. "C7;C6;C5")
 *   packstar.design.epsilon       — approximation ratio (default 0.683)
 *   packstar.design.numCPUs       — number of CPUs (default 2)
 *   packstar.design.designId      — design identifier for output
 *   packstar.design.outputDir     — output directory
 *
 */
public class PackStarDesignRunner {
    private static final class ConfSpaces {
        ForcefieldParams ffparams;
        SimpleConfSpace protein;
        SimpleConfSpace ligand;
        SimpleConfSpace complex;
    }

    public static void main(String[] args) throws Exception {
        // Read design spec from system properties
        String pdbPath = System.getProperty("packstar.design.pdbPath");
        String proteinChainsProp = System.getProperty("packstar.design.proteinChains", "");
        String ligandChainsProp = System.getProperty("packstar.design.ligandChains", "");
        String mutableStr = System.getProperty("packstar.design.mutable", "");
        String flexibleStr = System.getProperty("packstar.design.flexible", "");
        double epsilon = Double.parseDouble(System.getProperty("packstar.design.epsilon", "0.683"));
        int cpus = Integer.getInteger("packstar.design.numCPUs", 2);
        String designId = System.getProperty("packstar.design.designId", "unknown");
        String outputDir = System.getProperty("packstar.design.outputDir", "packstar-results");

        String[] mutableResidues = mutableStr.split(";");
        String[] flexibleResidues = flexibleStr.isEmpty() ? new String[0] : flexibleStr.split(";");

        System.out.println("==============================================");
        System.out.println("  PACK* Design");
        System.out.println("  Design: " + designId);
        System.out.println("  PDB: " + pdbPath);
        System.out.println("  Mutable: " + mutableStr);
        System.out.println("  Flexible: " + flexibleStr);
        System.out.println("  Epsilon: " + epsilon + ", CPUs: " + cpus);
        System.out.println("==============================================");

        if (pdbPath == null || pdbPath.isEmpty()) {
            System.err.println("ERROR: packstar.design.pdbPath is required");
            System.exit(1);
        }

        // Build conf spaces
        ForcefieldParams ffparams = new ForcefieldParams();
        Molecule mol = PDBIO.readFile(pdbPath);
        ResidueTemplateLibrary templateLib = new ResidueTemplateLibrary.Builder(ffparams.forcefld).build();

        // Standard amino-acid choices for an unrestricted mutable position.
        String[] all20 = {"ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY",
                "HIS","ILE","LEU","LYS","MET","PHE","PRO","SER",
                "THR","TRP","TYR","VAL"};

        // Detect all chain ranges
        Map<String, String[]> chainRanges = getChainRanges(mol);
        System.out.println("  Chains detected: " + chainRanges.keySet());

        Set<String> proteinChainSet = parseChainSet(proteinChainsProp);
        if (proteinChainSet.isEmpty()) {
            for (String res : mutableResidues) {
                res = res.trim();
                if (!res.isEmpty() && res.length() >= 2) proteinChainSet.add(String.valueOf(res.charAt(0)));
            }
        }
        Set<String> ligandChainSet = parseChainSet(ligandChainsProp);
        if (ligandChainSet.isEmpty()) {
            for (String ch : chainRanges.keySet()) {
                if (!proteinChainSet.contains(ch) && !ch.trim().isEmpty()) ligandChainSet.add(ch);
            }
        }
        System.out.println("  Protein chains: " + proteinChainSet);
        System.out.println("  Ligand chains: " + ligandChainSet);

        // Build one protein strand per chain so multi-chain proteins do not
        // accidentally swallow ligand/intermediate chains in PDB order.
        List<Strand> proteinStrands = new ArrayList<>();
        for (String ch : proteinChainSet) {
            String[] r = chainRanges.get(ch);
            if (r == null) {
                System.err.println("  WARNING: requested protein chain " + ch + " not detected in PDB");
                continue;
            }
            System.out.println("  Protein strand: " + r[0] + " - " + r[1]);
            proteinStrands.add(new Strand.Builder(mol).setTemplateLibrary(templateLib)
                    .setResidues(r[0], r[1]).build());
        }
        if (proteinStrands.isEmpty()) {
            throw new IllegalArgumentException("no protein strands could be built from protein chains " + proteinChainSet);
        }

        // Apply mutable residues to protein strand.
        // Token syntax: "<chain><resnum>" => all 20 AAs (back-compat);
        //               "<chain><resnum>=A,S,D" => only the listed AAs (1-letter) + wild type.
        for (String tok : mutableResidues) {
            tok = tok.trim(); if (tok.isEmpty()) continue;
            String res; String[] rotamers;
            int eq = tok.indexOf('=');
            if (eq >= 0) {
                res = tok.substring(0, eq).trim();
                rotamers = parseAAList(tok.substring(eq + 1));
            } else {
                res = tok; rotamers = all20;
            }
            if (applyResidueFlex(proteinStrands, res, rotamers)) {
                System.out.println("  Protein mutable: " + res + " -> " + String.join(",", rotamers));
            } else {
                System.err.println("  WARNING: mutable residue " + res + " not on protein strand");
            }
        }
        // Apply protein-side flexible residues
        for (String res : flexibleResidues) {
            res = res.trim(); if (res.isEmpty()) continue;
            if (!proteinChainSet.contains(String.valueOf(res.charAt(0)))) continue;
            if (applyResidueFlex(proteinStrands, res, Strand.WildType)) {
                System.out.println("  Protein flexible: " + res);
            } else {
                System.err.println("  WARNING: flexible residue " + res + " not on protein strand");
            }
        }

        // Assemble conf spaces
        ConfSpaces confSpaces = new ConfSpaces();
        confSpaces.ffparams = ffparams;

        if (!ligandChainSet.isEmpty()) {
            // Build one ligand strand per chain so we don't accidentally include
            // protein residues that lie between ligand chains in PDB ordering.
            List<Strand> ligandStrands = new ArrayList<>();
            for (String ch : ligandChainSet) {
                String[] r = chainRanges.get(ch);
                if (r == null) continue;
                System.out.println("  Ligand strand: " + r[0] + " - " + r[1]);
                Strand ligStrand = new Strand.Builder(mol).setTemplateLibrary(templateLib)
                        .setResidues(r[0], r[1]).build();

                // Apply ligand-side flexible residues for this chain only
                for (String res : flexibleResidues) {
                    res = res.trim(); if (res.isEmpty()) continue;
                    if (!ch.equals(String.valueOf(res.charAt(0)))) continue;
                    if (applyResidueFlex(Collections.singletonList(ligStrand), res, Strand.WildType)) {
                        System.out.println("  Ligand flexible: " + res);
                    } else {
                        System.err.println("  WARNING: flexible residue " + res + " not on ligand strand");
                    }
                }
                ligandStrands.add(ligStrand);
            }

            confSpaces.protein = new SimpleConfSpace.Builder().addStrands(proteinStrands).build();
            confSpaces.ligand = new SimpleConfSpace.Builder().addStrands(ligandStrands).build();
            SimpleConfSpace.Builder complexBuilder = new SimpleConfSpace.Builder().addStrands(proteinStrands);
            complexBuilder.addStrands(ligandStrands);
            confSpaces.complex = complexBuilder.build();
        } else {
            System.out.println("  WARNING: no ligand chains — K* will be trivial");
            confSpaces.protein = new SimpleConfSpace.Builder().addStrands(proteinStrands).build();
            confSpaces.ligand = new SimpleConfSpace.Builder().build();
            confSpaces.complex = confSpaces.protein;
        }

        System.out.println("  Protein positions: " + confSpaces.protein.positions.size());
        System.out.println("  Ligand positions: " + confSpaces.ligand.positions.size());
        System.out.println("  Complex positions: " + confSpaces.complex.positions.size());

        int numGpus = Integer.getInteger("packstar.design.numGpus", 0);
        Parallelism parallelism = numGpus > 0
                ? Parallelism.make(cpus, numGpus, Integer.getInteger("packstar.design.streamsPerGpu", 64))
                : Parallelism.makeCpu(cpus);
        String ematDir = outputDir + "/emat_cache/" + designId;
        new File(ematDir).mkdirs();

        long t0 = System.currentTimeMillis();

        runPackStar(confSpaces, epsilon, parallelism, ematDir, designId, outputDir, "packstar");

        long elapsed = System.currentTimeMillis() - t0;
        System.out.println("\n=== TOTAL TIME: " + String.format("%.1f", elapsed / 1000.0) + " s ===");
    }

    /**
     * Parse a comma-separated list of AA codes into OSPREY residue template names.
     * Accepts standard 1-letter codes (A,S,D,...) AND already-3-letter OSPREY
     * template names (HID,HIE,HIP,...) passed through unchanged/uppercased --
     * this lets callers request alternate-protonation-state templates (which
     * exist in all_amino94.in + LovellRotamer.dat for HIS tautomers) without
     * being silently collapsed to their 1-letter parent by aa1to3().
     */
    private static String[] parseAAList(String csv) {
        String[] parts = csv.split(",");
        List<String> out = new ArrayList<>();
        for (String p : parts) {
            p = p.trim();
            if (p.isEmpty()) continue;
            out.add(p.length() > 1 ? p.toUpperCase() : aa1to3(p.charAt(0)));
        }
        return out.toArray(new String[0]);
    }

    private static String aa1to3(char c) {
        switch (Character.toUpperCase(c)) {
            case 'A': return "ALA"; case 'R': return "ARG"; case 'N': return "ASN";
            case 'D': return "ASP"; case 'C': return "CYS"; case 'E': return "GLU";
            case 'Q': return "GLN"; case 'G': return "GLY"; case 'H': return "HIS";
            case 'I': return "ILE"; case 'L': return "LEU"; case 'K': return "LYS";
            case 'M': return "MET"; case 'F': return "PHE"; case 'P': return "PRO";
            case 'S': return "SER"; case 'T': return "THR"; case 'W': return "TRP";
            case 'Y': return "TYR"; case 'V': return "VAL";
            default: throw new IllegalArgumentException("Unknown 1-letter AA code: " + c);
        }
    }

    private static void runPackStar(ConfSpaces confSpaces, double epsilon,
                                    Parallelism parallelism, String ematDir,
                                    String designId, String outputDir,
                                    String outputMethodName) {

        EnergyCalculator.Type ecalcType = Integer.getInteger("packstar.design.numGpus", 0) > 0
                ? EnergyCalculator.Type.ResidueCudaCCD : EnergyCalculator.Type.Cpu;
        try (EnergyCalculator minimizingEcalc = new EnergyCalculator.Builder(
                confSpaces.complex, confSpaces.ffparams)
                .setParallelism(parallelism).setType(ecalcType).build();
        EnergyCalculator rigidEcalc = new EnergyCalculator.Builder(
                confSpaces.complex, confSpaces.ffparams)
                .setParallelism(parallelism).setType(ecalcType).setIsMinimizing(false).build()) {
            KStar.Settings settings = new KStar.Settings.Builder()
                    .setEpsilon(epsilon)
                    .setStabilityThreshold(null)
                    .setMaxSimultaneousMutations(1)
                    .setShowPfuncProgress(true)
                    .build();
            KStar kstar = new KStar(confSpaces.protein, confSpaces.ligand,
                    confSpaces.complex, settings);

            for (KStar.ConfSpaceInfo info : kstar.confSpaceInfos()) {
                SimpleConfSpace cs = (SimpleConfSpace) info.confSpace;
                String stateName = info.type.name();
                String cachePrefix = ematDir + "/packstar." + stateName.toLowerCase(Locale.ROOT);

                ConfEnergyCalculator minimizingConfEcalc = new ConfEnergyCalculator.Builder(cs, minimizingEcalc)
                        .setReferenceEnergies(new SimplerEnergyMatrixCalculator.Builder(cs, minimizingEcalc)
                                .build().calcReferenceEnergies())
                        .build();
                ConfEnergyCalculator rigidConfEcalc = new ConfEnergyCalculator.Builder(cs, rigidEcalc)
                        .setReferenceEnergies(new SimplerEnergyMatrixCalculator.Builder(cs, rigidEcalc)
                                .build().calcReferenceEnergies())
                        .build();
                info.confEcalc = minimizingConfEcalc;

                EnergyMatrix rigidEmat = new SimplerEnergyMatrixCalculator.Builder(rigidConfEcalc)
                        .setCacheFile(new File(cachePrefix + ".rigid.dat"))
                        .build().calcEnergyMatrix();
                EnergyMatrix minimizingEmat = new SimplerEnergyMatrixCalculator.Builder(minimizingConfEcalc)
                        .setCacheFile(new File(cachePrefix + ".minimizing.dat"))
                        .build().calcEnergyMatrix();

                info.pfuncFactory = (rcs) -> {
                    PackStarPartitionFunction pfunc = new PackStarPartitionFunction(
                            cs, rigidEmat, minimizingEmat, minimizingConfEcalc,
                            rcs, parallelism, stateName);

                    return pfunc;
                };
                info.confDBFile = null;
            }

            long packT0 = System.currentTimeMillis();
            List<KStar.ScoredSequence> scores = kstar.run(minimizingEcalc.tasks);
            minimizingEcalc.tasks.waitForFinish();
            rigidEcalc.tasks.waitForFinish();
            double packElapsed = (System.currentTimeMillis() - packT0) / 1000.0;
            writeKStarResults(scores, designId, outputMethodName, outputDir, epsilon, packElapsed);
        }
    }

    /** CSV schema for PACK* binding-score and state intervals. */
    private static final String RESULTS_HEADER =
        "rank,sequence,method,target_eps,score_log10,lb_log10,ub_log10," +
        "prot_qstar_lb_log10,prot_qstar_ub_log10,prot_status,prot_eps,prot_nconf," +
        "prot_nscored,prot_npartial," +
        "lig_qstar_lb_log10,lig_qstar_ub_log10,lig_status,lig_eps,lig_nconf," +
        "lig_nscored,lig_npartial," +
        "comp_qstar_lb_log10,comp_qstar_ub_log10,comp_status,comp_eps,comp_nconf," +
        "comp_nscored,comp_npartial," +
        "total_time_s";

    /** Format one row from a KStarScore + sequence index. */
    private static String formatScoreRow(int rank, Sequence sequence,
                                          edu.duke.cs.osprey.kstar.KStarScore score,
                                          String method, double targetEps, double totalTimeS) {
        Double scoreLog = score.scoreLog10();  // null when not Estimated
        String scoreStr = (scoreLog == null || scoreLog.isNaN()) ? "" : String.format("%.6f", scoreLog);
        Double lbLog = score.lowerBoundLog10();
        Double ubLog = score.upperBoundLog10();
        String lbStr = (lbLog == null || lbLog.isNaN()) ? "" : String.format("%.6f", lbLog);
        String ubStr = (ubLog == null || ubLog.isNaN()) ? "" : String.format("%.6f", ubLog);
        return String.format("%d,%s,%s,%.6f,%s,%s,%s,%s,%s,%s,%.1f",
            rank,
            sequence.toString(Sequence.Renderer.ResType),
            method,
            targetEps,
            scoreStr, lbStr, ubStr,
            formatPfunc(score.protein),
            formatPfunc(score.ligand),
            formatPfunc(score.complex),
            totalTimeS);
    }

    /**
     * Returns seven columns for one pfunc result:
     * "qstar_lb_log10,qstar_ub_log10,status,eps,numConfs,
     *  nscored,npartial"
     */
    private static String formatPfunc(edu.duke.cs.osprey.kstar.pfunc.PartitionFunction.Result r) {
        if (r == null) return ",,N/A,,0,0,0";
        Double lb = edu.duke.cs.osprey.kstar.KStarScore.scoreToLog10(r.values.calcLowerBound());
        Double ub = edu.duke.cs.osprey.kstar.KStarScore.scoreToLog10(r.values.calcUpperBound());
        String lbStr = (lb == null || lb.isNaN()) ? "" : String.format("%.6f", lb);
        String ubStr = (ub == null || ub.isNaN()) ? "" : String.format("%.6f", ub);
        String epsStr = "";
        try {
            double eps = r.values.getEffectiveEpsilon();
            epsStr = Double.isNaN(eps) || Double.isInfinite(eps) ? "" : String.format("%.6f", eps);
        } catch (RuntimeException e) {
            // MARK*/PackStar bounds can use MagicBigDecimal infinities; keep the
            // CSV row writable even when epsilon is not numerically meaningful.
        }
        return String.format("%s,%s,%s,%s,%d,%d,%d",
            lbStr, ubStr, r.status.name(), epsStr, r.numConfs,
            r.getStat("numConfsScored"), r.getStat("numPartialMinimizations"));
    }

    private static void writeKStarResults(List<KStar.ScoredSequence> scores,
                                           String designId, String method, String outputDir,
                                           double targetEps, double totalTimeS) {
        String csvPath = outputDir + "/" + designId + "_" + method + ".csv";
        try (PrintWriter pw = new PrintWriter(new FileWriter(csvPath))) {
            pw.println(RESULTS_HEADER);
            scores.sort((a, b) -> Double.compare(b.score.lowerBoundLog10(), a.score.lowerBoundLog10()));
            for (int i = 0; i < scores.size(); i++) {
                KStar.ScoredSequence s = scores.get(i);
                pw.println(formatScoreRow(i + 1, s.sequence, s.score, method, targetEps, totalTimeS));
            }
            System.out.println("Results written to " + csvPath);
        } catch (IOException e) {
            System.err.println("Error writing CSV: " + e.getMessage());
        }
    }

    private static Set<String> parseChainSet(String chains) {
        Set<String> out = new LinkedHashSet<>();
        for (String token : chains.split(",")) {
            token = token.trim();
            if (!token.isEmpty()) out.add(token);
        }
        return out;
    }

    private static boolean applyResidueFlex(List<Strand> strands, String res, String... rotamers) {
        for (Strand strand : strands) {
            var flex = strand.flexibility.get(res);
            if (flex == null) continue;
            flex.setLibraryRotamers(rotamers).addWildTypeRotamers().setContinuous();
            return true;
        }
        return false;
    }

    private static Map<String, String[]> getChainRanges(Molecule mol) {
        Map<String, String[]> ranges = new LinkedHashMap<>();
        Map<String, String> firstRes = new LinkedHashMap<>();
        Map<String, String> lastRes = new LinkedHashMap<>();
        for (var res : mol.residues) {
            String chain = String.valueOf(res.getChainId());
            String fullId = res.getPDBResNumber().trim();
            if (!firstRes.containsKey(chain)) firstRes.put(chain, fullId);
            lastRes.put(chain, fullId);
        }
        for (String ch : firstRes.keySet()) {
            ranges.put(ch, new String[]{firstRes.get(ch), lastRes.get(ch)});
        }
        return ranges;
    }
}
