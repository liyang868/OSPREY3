package edu.duke.cs.osprey.packstar;

import edu.duke.cs.osprey.astar.conf.RCs;

import edu.duke.cs.osprey.branchdp.BranchDpConfig;

/**
 * Shared PACK* system-property resolver.
 *
 * <p>PACK* owns the {@code packstar.*} keys. When a shared branch-DP key is
 * passed in, the matching PACK* key is preferred and the neutral
 * {@code branchdp.*} key remains available as a fallback.</p>
 */
public final class PackStarConfig {

    private PackStarConfig() {
    }

    public static String getProperty(String key, String defaultValue) {
        return getProperty(key, defaultValue, true);
    }

    public static String getProperty(String key, String defaultValue, boolean preferPackStarAliases) {
        return BranchDpConfig.getProperty(key, defaultValue, preferPackStarAliases);
    }

    public static boolean getBoolean(String key, boolean defaultValue) {
        return getBoolean(key, defaultValue, true);
    }

    public static boolean getBoolean(String key, boolean defaultValue, boolean preferPackStarAliases) {
        String value = getProperty(key, null, preferPackStarAliases);
        return value == null ? defaultValue : Boolean.parseBoolean(value);
    }

    public static int getInteger(String key, int defaultValue, String warningPrefix) {
        return getInteger(key, defaultValue, warningPrefix, true);
    }

    public static int getInteger(String key, int defaultValue, String warningPrefix,
                                 boolean preferPackStarAliases) {
        String value = getProperty(key, null, preferPackStarAliases);
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

    public static long getLong(String key, long defaultValue, String warningPrefix) {
        return getLong(key, defaultValue, warningPrefix, true);
    }

    public static long getLong(String key, long defaultValue, String warningPrefix,
                               boolean preferPackStarAliases) {
        String value = getProperty(key, null, preferPackStarAliases);
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

    public static double getDouble(String key, double defaultValue, String warningPrefix) {
        return getDouble(key, defaultValue, warningPrefix, true);
    }

    public static double getDouble(String key, double defaultValue, String warningPrefix,
                                   boolean preferPackStarAliases) {
        String value = getProperty(key, null, preferPackStarAliases);
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

    private static void warn(String warningPrefix, String kind, String key, String value, String defaultValue) {
        System.err.println(warningPrefix + " " + kind + " for '" + key
                + "': '" + value + "', using " + defaultValue + ".");
    }

    public static String stateKey(String stateName, RCs rcs) {
        StringBuilder key = new StringBuilder(
                stateName == null ? "unknown" : stateName);
        key.append('|').append(rcs.getNumPos());
        for (int pos = 0; pos < rcs.getNumPos(); pos++) {
            key.append('|').append(pos).append(':');
            int[] allowed = rcs.get(pos);
            for (int rc : allowed) {
                key.append(rc).append(',');
            }
        }
        return key.toString();
    }
}
