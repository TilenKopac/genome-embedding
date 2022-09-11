from enum import Enum


class SamplerEnum(Enum):
    NO_SAMPLER = "no-sampler"
    CENTROID = "centroid-sampler"
    RANDOM = "random-sampler"
    CONVEX_HULL = "convex-hull-sampler"
    HYPERCUBE_FINGERPRINT_MEDIAN = "hypercube-fingerprint-median-sampler"
    HYPERCUBE_FINGERPRINT_MEDIAN_NORMALIZED = "hypercube-fingerprint-median-normalized-sampler"
    HYPERCUBE_FINGERPRINT_QUARTILE = "hypercube-fingerprint-quartile-sampler"
    HYPERCUBE_FINGERPRINT_QUARTILE_NORMALIZED = "hypercube-fingerprint-quartile-normalized-sampler"
    HYPERCUBE_FINGERPRINT_OCTILE = "hypercube-fingerprint-octile-sampler"
    HYPERCUBE_FINGERPRINT_OCTILE_NORMALIZED = "hypercube-fingerprint-octile-normalized-sampler"
