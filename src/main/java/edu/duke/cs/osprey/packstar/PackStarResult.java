package edu.duke.cs.osprey.packstar;

import edu.duke.cs.osprey.kstar.pfunc.PartitionFunction;

/**
 * PACK* result retained by ordinary K* result caches and score writers.
 */
public final class PackStarResult extends PartitionFunction.Result {

    /** Actual full-conformation CCD calls, also reported by numConfs. */
    public static final String FULL_CCD_STAT = "fullCCD";
    /** Logical draws submitted to CCD, including duplicates and cache hits. */
    public static final String CCD_SAMPLE_RECORDS_STAT = "ccdSampleRecords";

    PackStarResult(PartitionFunction.Result source) {
        super(source.status, source.values, source.numConfs);
        source.getStats().forEach(this::setStat);
    }
}
