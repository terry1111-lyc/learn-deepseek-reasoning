"""Idealized cache arithmetic, not attention or a deployment estimator.

Dimensions come from the pinned V3 config in SOURCES.md. Assumes identical
bytes per scalar, unsharded caches, and the naive expanded representation.
Excludes weights, metadata, padding, activations and allocator overhead.
"""


def cache_sizes(tokens=4096, layers=61, batch=1, bytes_per_scalar=2,
                heads=128, nope_dim=128, rope_dim=64, value_dim=128,
                latent_dim=512):
    values = (tokens, layers, batch, bytes_per_scalar, heads,
              nope_dim, rope_dim, value_dim, latent_dim)
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError("All dimensions must be positive integers")
    factor = tokens * layers * batch * bytes_per_scalar
    naive = factor * heads * (nope_dim + rope_dim + value_dim)
    mla = factor * (latent_dim + rope_dim)
    return naive, mla


if __name__ == "__main__":
    naive, mla = cache_sizes()
    print("Hypothetical unsharded BF16 cache: batch=1, tokens=4096, layers=61")
    print(f"naive expanded = {naive / 2**30:.4f} GiB")
    print(f"MLA absorb     = {mla / 2**30:.4f} GiB")
    print(f"element ratio  = {naive / mla:.4f}x (NOT speedup)")
