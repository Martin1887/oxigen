# Instructions to migrate between Oxigen major versions

## Migrate from Oxigen 1.x to 2.x
- Instead of implementing `FromIterator<T>` in your `Genotype<T>` struct, write a `from_iter<I: Iterator<Item = T>>(&mut self, genes: I)` function in the `Genotype` trait implementation that set the genes collecting the iterator:
```rust
// In 1.x versions
impl FromIterator<u8> for QueensBoard {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = u8>,
    {
        QueensBoard {
            0: iter.into_iter().collect(),
        }
    }
}

// In 2.x versions
impl Genotype<u8> for QueensBoard {
    // [...]
    fn from_iter<I: Iterator<Item = u8>>(&mut self, genes: I) {
        self.0 = genes.collect();
    }
    // [...]
}
```
- Return a boolean in the `fix` function specifying if the individual has been changed.
- Replace the `Cuadratic` enum variants by `Quadratic`.