package edu.duke.cs.osprey.packstar;

import edu.duke.cs.osprey.kstar.pfunc.PartitionFunction;
import edu.duke.cs.osprey.confspace.ConfSearch;

/**
 * PACK* result retained by ordinary K* result caches and score writers.
 */
public final class PackStarResult extends PartitionFunction.Result {

    /** Actual full-conformation CCD calls, also reported by numConfs. */
    public static final String FULL_CCD_STAT = "fullCCD";
    /** Logical draws submitted to CCD, including duplicates and cache hits. */
    public static final String CCD_SAMPLE_RECORDS_STAT = "ccdSampleRecords";

    /** Final proposal GMEC, or null when disabled or no finite validated proposal exists. */
    public final ConfSearch.ScoredConf gmec;

    PackStarResult(PartitionFunction.Result source) {
        this(source, null);
    }

    PackStarResult(PartitionFunction.Result source, ConfSearch.ScoredConf gmec) {
        super(source.status, source.values, source.numConfs);
        source.getStats().forEach(this::setStat);
        this.gmec = gmec;
    }
}
