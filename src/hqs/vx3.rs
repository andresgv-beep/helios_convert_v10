//! HQS x.3 (hq43k_g16 / hq53k_g16): códec de GPT, copiado tal cual de helios-convert-hqs-vx3
//! (28/09/2026). Lo usa hqs_v4_convert para la atención y el embedding; su salida coincide
//! byte a byte con la del HNF v4 validado.
//! Experimental x.3 codec. Group size is part of the dtype; no production IDs.
use anyhow::{ensure, Result};
use half::f16;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct Format { bits: usize, group: usize }
impl Format {
    pub fn new(bits: usize, group: usize) -> Result<Self> {
        ensure!((3..=5).contains(&bits), "x.3 requires 3, 4 or 5 bits");
        ensure!(group == 16 || group == 32, "x.3 requires group 16 or 32");
        Ok(Self { bits, group })
    }
    pub fn header(self) -> usize { (2 + 256 / self.group * 5 / 8 + 3) / 4 * 4 }
    pub fn block_bytes(self) -> usize { self.header() + self.bits * 32 }
    pub fn dtype(self) -> String { format!("hq{}3k_g{}", self.bits, self.group) }
    pub fn quantize(self, values: &[f32]) -> Result<Vec<u8>> {
        ensure!(values.iter().all(|x| x.is_finite()), "non-finite weights");
        let blocks = (values.len() + 255) / 256;
        let mut out = vec![0u8; blocks * self.block_bytes()];
        // Validate FP16 scale range before parallel workers produce any bytes.
        let bound = values.iter().fold(0f32, |m,x| m.max(x.abs())) / ((1 << (self.bits-1))-1) as f32;
        ensure!(bound <= 65504., "weights exceed FP16 scale range");
        out.par_chunks_mut(self.block_bytes()).enumerate().for_each(|(i, dst)| {
            let mut block = [0f32; 256];
            let src = &values[i*256..values.len().min((i+1)*256)];
            block[..src.len()].copy_from_slice(src);
            self.encode_block(&block, dst);
        });
        Ok(out)
    }
    fn encode_block(self, x: &[f32;256], out: &mut [u8]) {
        let negative = (1 << (self.bits-1)) as f32;
        let positive = negative - 1.;
        let groups = 256 / self.group;
        let mut ideal = [0f32;16];
        for g in 0..groups {
            let values = &x[g*self.group..(g+1)*self.group];
            let pos = values.iter().copied().fold(0.,f32::max);
            let neg = -values.iter().copied().fold(0.,f32::min);
            ideal[g] = (pos / positive).max(neg / negative);
        }
        let dh = f16::from_f32(ideal.iter().copied().fold(0., f32::max));
        let d = dh.to_f32(); out[..2].copy_from_slice(&dh.to_le_bytes());
        for g in 0..groups {
            let q0 = (ideal[g] / d.max(1e-30) * 31.).round().clamp(1.,31.) as i32;
            let mut best = f32::INFINITY; let mut best_qs = 0u8; let mut best_codes = [0u8;32];
            for delta in -3..=3 {
                let qs = (q0 + delta).clamp(1,31) as u8;
                let step = d * qs as f32 * (1. / 31.);
                let mut error = 0f32; let mut codes = [0u8;32];
                for j in 0..self.group {
                    let v = x[g*self.group+j];
                    let q = (v / step.max(1e-30)).round().clamp(-negative,positive);
                    let diff = v - q*step; error += diff*diff;
                    codes[j] = (q + negative) as u8;
                }
                if error < best { best = error; best_qs = qs; best_codes = codes; }
            }
            put_bits(&mut out[2..self.header()],g,5,best_qs);
            for j in 0..self.group {
                let index = g*self.group+j; let code = best_codes[j];
                if self.bits == 4 {
                    out[self.header()+index/2] |= code << if index%2==0 {4} else {0};
                } else { put_bits(&mut out[self.header()..],index,self.bits,code); }
            }
        }
    }
    pub fn dequantize(self, bytes: &[u8], count: usize) -> Result<Vec<f32>> {
        ensure!(bytes.len() == (count+255)/256*self.block_bytes(), "invalid x.3 byte count");
        let mut out = Vec::with_capacity(count);
        for block in bytes.chunks_exact(self.block_bytes()) {
            let d = f16::from_le_bytes([block[0],block[1]]).to_f32();
            for i in 0..256 {
                if out.len() == count { break; }
                let qs = get_bits(&block[2..self.header()],i/self.group,5);
                let step = d * qs as f32 * (1./31.);
                let q = if self.bits == 4 { (block[self.header()+i/2] >> if i%2==0 {4} else {0}) & 15 }
                        else { get_bits(&block[self.header()..],i,self.bits) };
                out.push((q as i32 - (1 << (self.bits-1))) as f32 * step);
            }
        }
        Ok(out)
    }
}
fn put_bits(out: &mut [u8], i: usize, bits: usize, value: u8) {
    let p=i*bits; let byte=p/8; let shift=p%8; let v=(value as u16)<<shift;
    out[byte] |= v as u8;
    if shift+bits>8 { out[byte+1] |= (v>>8) as u8; }
}
fn get_bits(data: &[u8], i: usize, bits: usize) -> u8 {
    let p=i*bits; let byte=p/8; let shift=p%8;
    let v=data[byte] as u16 | ((data.get(byte+1).copied().unwrap_or(0) as u16)<<8);
    ((v>>shift)&((1<<bits)-1)) as u8
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn partial_zero_constant_and_invalid() {
        for bits in 3..=5 {for group in [16,32] {
            let f=Format::new(bits,group).unwrap();
            for n in [0,1,255,256,257,1024] {
                for value in [0.,0.125,-0.125] {
                    let x=vec![value;n];let q=f.quantize(&x).unwrap();let y=f.dequantize(&q,n).unwrap();
                    assert_eq!(y.len(),n);assert!(y.iter().all(|v|v.is_finite()));
                    if value==0. {assert!(y.iter().all(|&v|v==0.));}
                }
            }
            assert!(f.quantize(&[f32::NAN]).is_err());assert!(f.quantize(&[f32::INFINITY]).is_err());
            assert!(f.dequantize(&[0;3],256).is_err());
        }}
        assert!(Format::new(2,16).is_err());assert!(Format::new(4,8).is_err());
    }
    #[test] fn cross_language_original_weight_fixtures() {
        let root=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("lab/fixtures");
        if !root.join("manifest.json").exists() { return; }
        let m:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
        for case in m.as_array().unwrap() {
            let stem=case["stem"].as_str().unwrap();
            let raw=std::fs::read(root.join(format!("{stem}.f32"))).unwrap();
            let x:Vec<f32>=raw.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
            let f=Format::new(case["bits"].as_u64().unwrap() as usize,case["group"].as_u64().unwrap() as usize).unwrap();
            let expected=std::fs::read(root.join(format!("{stem}.bin"))).unwrap();
            assert_eq!(f.quantize(&x).unwrap(),expected,"fixture {stem}");
        }
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;
    #[test]
    fn manual_little_endian_blocks_decode_without_encoder() {
        for bits in 3..=5 { for group in [16,32] {
            let f=Format::new(bits,group).unwrap();let mut block=vec![0u8;f.block_bytes()];
            block[..2].copy_from_slice(&f16::from_f32(1.).to_le_bytes());
            block[2..2+256/group*5/8].fill(255);
            block[f.header()..].fill(if bits==4 {0xf0} else {0xff});
            let y=f.dequantize(&block,256).unwrap();
            for (i,&v) in y.iter().enumerate() {
                let expected=if bits==4 {if i%2==0 {7.} else {-8.}} else {((1<<(bits-1))-1) as f32};
                assert_eq!(v,expected);
            }
        }}
    }
}
