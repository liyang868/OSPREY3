package edu.duke.cs.osprey.branchdp;

import java.util.Locale;

/**
 * Shared branch-DP system-property resolver.
 *
 * <p>The branch-DP primitives own the neutral {@code branchdp.*} keys.
 * PACK* callers may enter a scoped alias mode where matching {@code packstar.*}
 * keys are preferred.</p>
 */
public final class BranchDpConfig {

    private static final ThreadLocal<Integer> PACKSTAR_ALIAS_DEPTH =
            ThreadLocal.withInitial(() -> 0);

    private BranchDpConfig() {
    }

    public static boolean isPackStarAliasScopeActive() {
        return PACKSTAR_ALIAS_DEPTH.get() > 0;
    }

    public static String getBackendName() {
        return isPackStarAliasScopeActive() ? "PACK*" : "Branch-DP";
    }

    public static String getBackendLogPrefix() {
        return getBackendName() + ":";
    }

    public static String getBackendThreadNamePrefix() {
        return isPackStarAliasScopeActive() ? "packstar" : "branchdp";
    }

    public static Scope enterPackStarAliasScope() {
        PACKSTAR_ALIAS_DEPTH.set(PACKSTAR_ALIAS_DEPTH.get() + 1);
        return new Scope();
    }

    public static String getBackendProperty(String key, String defaultValue) {
        return getProperty(key, defaultValue, isPackStarAliasScopeActive());
    }

    public static boolean getBackendBoolean(String key, boolean defaultValue) {
        String value = getBackendProperty(key, null);
        return value == null ? defaultValue : Boolean.parseBoolean(value);
    }

    public static int getBackendInteger(String key, int defaultValue, String warningPrefix) {
        String value = getBackendProperty(key, null);
        if (value == null || value.trim().isEmpty()) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            warn(warningPrefix, "Invalid integer", key, value, Integer.toString(defaultValue));
            return defaultValue;
        }
    }

    public static long getBackendLong(String key, long defaultValue, String warningPrefix) {
        String value = getBackendProperty(key, null);
        if (value == null || value.trim().isEmpty()) {
            return defaultValue;
        }
        try {
            return Long.parseLong(value.trim());
        } catch (NumberFormatException e) {
            warn(warningPrefix, "Invalid long", key, value, Long.toString(defaultValue));
            return defaultValue;
        }
    }

    public static double getBackendDouble(String key, double defaultValue, String warningPrefix) {
        String value = getBackendProperty(key, null);
        if (value == null || value.trim().isEmpty()) {
            return defaultValue;
        }
        try {
            return Double.parseDouble(value.trim());
        } catch (NumberFormatException e) {
            warn(warningPrefix, "Invalid double", key, value, Double.toString(defaultValue));
            return defaultValue;
        }
    }

    public static long getBackendBytes(String key, long defaultValue, String warningPrefix) {
        String value = getBackendProperty(key, null);
        if (value == null || value.trim().isEmpty()) {
            return defaultValue;
        }
        try {
            return parseByteCount(value.trim());
        } catch (NumberFormatException e) {
            warn(warningPrefix, "Invalid byte count", key, value, Long.toString(defaultValue));
            return defaultValue;
        }
    }

    /** Resolve the public PACK* keys and internal branch-DP keys. */
    public static String getProperty(String key, String defaultValue,
                                     boolean preferPackStarAliases) {
        if (preferPackStarAliases) {
            String preferredKey = preferPackStarKey(key);
            String value = directProperty(preferredKey);
            if (value == null) {
                String neutralKey = neutralBranchDpKey(preferredKey);
                if (!neutralKey.equals(preferredKey)) {
                    value = directProperty(neutralKey);
                }
            }
            return value != null ? value : defaultValue;
        }

        String value = directProperty(key);
        return value != null ? value : defaultValue;
    }

    private static String directProperty(String key) {
        return System.getProperty(key);
    }

    private static String preferPackStarKey(String key) {
        String prefix = "branchdp.";
        if (key.startsWith(prefix)) {
            return "packstar." + key.substring(prefix.length());
        }
        return key;
    }

    private static String neutralBranchDpKey(String key) {
        String prefix = "packstar.";
        if (key.startsWith(prefix)) {
            return "branchdp." + key.substring(prefix.length());
        }
        return key;
    }

    private static void warn(String warningPrefix, String kind, String key, String value, String defaultValue) {
        System.err.println(warningPrefix + " " + kind + " for '" + key
                + "': '" + value + "', using " + defaultValue + ".");
    }

    public static long parseByteCount(String value) {
        String normalized = value.trim().toLowerCase(Locale.ROOT).replace("_", "");
        long multiplier = 1L;
        if (normalized.endsWith("kib")) {
            multiplier = 1024L;
            normalized = normalized.substring(0, normalized.length() - 3);
        } else if (normalized.endsWith("kb") || normalized.endsWith("k")) {
            multiplier = 1024L;
            normalized = normalized.replaceAll("kb?$", "");
        } else if (normalized.endsWith("mib")) {
            multiplier = 1024L * 1024L;
            normalized = normalized.substring(0, normalized.length() - 3);
        } else if (normalized.endsWith("mb") || normalized.endsWith("m")) {
            multiplier = 1024L * 1024L;
            normalized = normalized.replaceAll("mb?$", "");
        } else if (normalized.endsWith("gib")) {
            multiplier = 1024L * 1024L * 1024L;
            normalized = normalized.substring(0, normalized.length() - 3);
        } else if (normalized.endsWith("gb") || normalized.endsWith("g")) {
            multiplier = 1024L * 1024L * 1024L;
            normalized = normalized.replaceAll("gb?$", "");
        } else if (normalized.endsWith("tib")) {
            multiplier = 1024L * 1024L * 1024L * 1024L;
            normalized = normalized.substring(0, normalized.length() - 3);
        } else if (normalized.endsWith("tb") || normalized.endsWith("t")) {
            multiplier = 1024L * 1024L * 1024L * 1024L;
            normalized = normalized.replaceAll("tb?$", "");
        } else if (normalized.endsWith("b")) {
            normalized = normalized.substring(0, normalized.length() - 1);
        }

        double amount = Double.parseDouble(normalized.trim());
        if (amount < 0 || amount > Long.MAX_VALUE / (double) multiplier) {
            throw new NumberFormatException(value);
        }
        return (long) (amount * multiplier);
    }

    public static final class Scope implements AutoCloseable {

        private boolean closed = false;

        private Scope() {
        }

        @Override
        public void close() {
            if (closed) {
                return;
            }
            int depth = PACKSTAR_ALIAS_DEPTH.get();
            if (depth <= 1) {
                PACKSTAR_ALIAS_DEPTH.remove();
            } else {
                PACKSTAR_ALIAS_DEPTH.set(depth - 1);
            }
            closed = true;
        }
    }
}
