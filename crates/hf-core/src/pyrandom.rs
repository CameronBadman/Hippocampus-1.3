//! `random.Random` as CPython 3.12 implements it — MT19937 seeded with
//! `init_by_array` over the seed's 32-bit words, `getrandbits`,
//! `_randbelow_with_getrandbits`, `randrange(stop)` and `sample`.
//!
//! The Python readers replay a training run's draws with
//! `random.Random(seed).sample(range(pool), microbatch)` per update and refuse
//! to read a run whose replayed histogram differs; a Rust runner that draws
//! its batches with this type keeps that replay exact. Only the operations the
//! engine uses are implemented; `random()` (the float path) is not.

use serde::{Deserialize, Serialize};

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// The generator's full state — what `getstate()` returns, minus the version
/// tag and the unused Gaussian cache.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PyRandomState {
    pub mt: Vec<u32>,
    pub index: usize,
}

#[derive(Clone, Debug)]
pub struct PyRandom {
    mt: [u32; N],
    index: usize,
}

impl PyRandom {
    /// `random.Random(seed)` for a non-negative integer seed given as
    /// little-endian 32-bit words (`seed_words_from_decimal` builds them).
    pub fn from_seed_words(key: &[u32]) -> Self {
        let key: Vec<u32> = if key.is_empty() {
            vec![0]
        } else {
            key.to_vec()
        };
        let mut rng = Self {
            mt: [0; N],
            index: N,
        };
        rng.init_genrand(19_650_218);
        let (mut i, mut j) = (1usize, 0usize);
        let mut k = N.max(key.len());
        while k > 0 {
            let prev = rng.mt[i - 1] ^ (rng.mt[i - 1] >> 30);
            rng.mt[i] = (rng.mt[i] ^ prev.wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                rng.mt[0] = rng.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        let mut k = N - 1;
        while k > 0 {
            let prev = rng.mt[i - 1] ^ (rng.mt[i - 1] >> 30);
            rng.mt[i] = (rng.mt[i] ^ prev.wrapping_mul(1_566_083_941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                rng.mt[0] = rng.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        rng.mt[0] = 0x8000_0000;
        rng.index = N;
        rng
    }

    /// `random.Random(seed)` for a seed that fits in 128 bits.
    pub fn from_seed(seed: u128) -> Self {
        Self::from_seed_words(&seed_words_from_u128(seed))
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            self.mt[i] = 1_812_433_253u32
                .wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        self.index = N;
    }

    /// One 32-bit output of the Mersenne Twister.
    pub fn genrand_u32(&mut self) -> u32 {
        if self.index >= N {
            for kk in 0..N {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[(kk + 1) % N] & LOWER_MASK);
                let mag = if y & 1 == 1 { MATRIX_A } else { 0 };
                self.mt[kk] = self.mt[(kk + M) % N] ^ (y >> 1) ^ mag;
            }
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `getrandbits(k)` as little-endian 32-bit words (CPython's `wordarray`).
    pub fn getrandbits_words(&mut self, k: u32) -> Vec<u32> {
        assert!(k > 0, "number of bits must be greater than zero");
        if k <= 32 {
            return vec![self.genrand_u32() >> (32 - k)];
        }
        let words = ((k - 1) / 32 + 1) as usize;
        let mut out = Vec::with_capacity(words);
        let mut k = k;
        for _ in 0..words {
            let mut r = self.genrand_u32();
            if k < 32 {
                r >>= 32 - k;
            }
            out.push(r);
            k = k.saturating_sub(32);
        }
        out
    }

    /// `getrandbits(k)` for `k <= 128`.
    pub fn getrandbits(&mut self, k: u32) -> u128 {
        assert!(k <= 128, "getrandbits: at most 128 bits through this path");
        let words = self.getrandbits_words(k);
        words
            .iter()
            .enumerate()
            .fold(0u128, |acc, (i, w)| acc | ((*w as u128) << (32 * i)))
    }

    /// `_randbelow_with_getrandbits(n)`: draw `n.bit_length()` bits, reject and redraw while `>= n`.
    pub fn randbelow(&mut self, n: u64) -> u64 {
        assert!(n > 0, "randbelow: n must be positive");
        let k = 64 - n.leading_zeros();
        loop {
            let r = self.getrandbits(k) as u64;
            if r < n {
                return r;
            }
        }
    }

    /// `randrange(stop)` for a positive stop.
    pub fn randrange(&mut self, stop: u64) -> u64 {
        assert!(stop > 0, "empty range for randrange()");
        self.randbelow(stop)
    }

    /// `sample(range(n), k)`: the indices CPython 3.12 draws, in draw order —
    /// the pool method for small populations, the rejection-into-a-set method
    /// otherwise, with the same `setsize` boundary.
    pub fn sample(&mut self, n: usize, k: usize) -> Vec<usize> {
        assert!(k <= n, "sample larger than population or is negative");
        let mut result = Vec::with_capacity(k);
        let mut setsize = 21usize;
        if k > 5 {
            let exponent = ((k as f64 * 3.0).ln() / 4f64.ln()).ceil() as u32;
            setsize += 4usize.pow(exponent);
        }
        if n <= setsize {
            let mut pool: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = self.randbelow((n - i) as u64) as usize;
                result.push(pool[j]);
                pool[j] = pool[n - i - 1];
            }
        } else {
            let mut selected = std::collections::HashSet::with_capacity(k);
            for _ in 0..k {
                let mut j = self.randbelow(n as u64) as usize;
                while selected.contains(&j) {
                    j = self.randbelow(n as u64) as usize;
                }
                selected.insert(j);
                result.push(j);
            }
        }
        result
    }

    /// `random()`: a float in [0, 1) from 53 bits, exactly as CPython builds it.
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as f64;
        let b = (self.genrand_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `uniform(a, b)`: `a + (b - a) * random()`.
    pub fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.random()
    }

    /// `getstate()`.
    pub fn state(&self) -> PyRandomState {
        PyRandomState {
            mt: self.mt.to_vec(),
            index: self.index,
        }
    }

    /// `setstate(state)`.
    pub fn from_state(state: &PyRandomState) -> Result<Self, crate::HfError> {
        if state.mt.len() != N || state.index > N {
            return Err(crate::HfError::Invalid(format!(
                "random state must carry {N} words and an index <= {N}"
            )));
        }
        let mut mt = [0u32; N];
        mt.copy_from_slice(&state.mt);
        Ok(Self {
            mt,
            index: state.index,
        })
    }
}

/// The little-endian 32-bit words of a non-negative seed, as CPython keys the
/// twister with them (a zero seed is the single word 0).
pub fn seed_words_from_u128(seed: u128) -> Vec<u32> {
    if seed == 0 {
        return vec![0];
    }
    let mut words = Vec::new();
    let mut s = seed;
    while s > 0 {
        words.push((s & 0xffff_ffff) as u32);
        s >>= 32;
    }
    words
}

/// The same words for a seed written in decimal of any length.
pub fn seed_words_from_decimal(text: &str) -> Result<Vec<u32>, crate::HfError> {
    let digits: Vec<u32> = text
        .trim()
        .chars()
        .map(|c| {
            c.to_digit(10)
                .ok_or_else(|| crate::HfError::Invalid(format!("not a decimal seed: {text:?}")))
        })
        .collect::<Result<_, _>>()?;
    if digits.is_empty() {
        return Err(crate::HfError::Invalid("empty seed".into()));
    }
    // repeated division of the decimal by 2^32, least significant word first
    let mut number = digits;
    let mut words = Vec::new();
    while !number.is_empty() && !number.iter().all(|d| *d == 0) {
        let mut remainder: u64 = 0;
        let mut quotient = Vec::with_capacity(number.len());
        for d in &number {
            let current = remainder * 10 + *d as u64;
            quotient.push((current >> 32) as u32);
            remainder = current & 0xffff_ffff;
        }
        words.push(remainder as u32);
        number = quotient.into_iter().skip_while(|q| *q == 0).collect();
    }
    if words.is_empty() {
        words.push(0);
    }
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_and_u128_seed_words_agree() {
        for seed in [0u128, 1, 1729, u32::MAX as u128, 1 << 32, (1 << 64) + 12345] {
            assert_eq!(
                seed_words_from_decimal(&seed.to_string()).unwrap(),
                seed_words_from_u128(seed)
            );
        }
    }

    #[test]
    fn state_round_trips() {
        let mut a = PyRandom::from_seed(1729);
        a.randrange(1000);
        let s = a.state();
        let mut b = PyRandom::from_state(&s).unwrap();
        assert_eq!(a.randrange(40000), b.randrange(40000));
        assert_eq!(a.sample(40000, 8), b.sample(40000, 8));
    }
}
