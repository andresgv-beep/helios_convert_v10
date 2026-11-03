//! Experimental payloads using adopted GM4X v2, not full-model HNF serialization.
//! Buffers one physical row; source callbacks provide logical columns only.
use anyhow::{ensure, Context, Result};
use half::{bf16, f16};
use std::io::Write;
use super::gemma4_moe_v2::{MatrixLayout, Storage};

/// Writes expert-major, then row-major data using the adopted GM4X v2 descriptor.
/// Gate/up rows must be supplied in that order WITHIN each expert.
///
/// The callback must fill every logical column. NaN sentinels detect missing data.
/// BF16 source values promoted to f32 are preserved exactly when writing BF16.
/// HQ44 is uncalibrated (unit importance), for layout/kernel experiments only.
/// On any failure the destination may be partial: callers must retain INCOMPLETE
/// until all payloads and their manifests are durable.
pub fn write_candidate_payload<W, F>(writer: &mut W, layout: &MatrixLayout, mut read_row: F) -> Result<u64>
where
    W: Write,
    F: FnMut(u32, u32, &mut [f32]) -> Result<()>,
{
    let logical = usize::try_from(layout.logical_cols())?;
    let physical = usize::try_from(layout.physical_cols())?;
    let mut row = Vec::new();
    row.try_reserve_exact(physical).context("row allocation failed")?;
    row.resize(physical, 0.0f32);
    for expert in 0..layout.experts() {
        for index in 0..layout.rows() {
            row[..logical].fill(f32::NAN);
            read_row(expert, index, &mut row[..logical])
                .with_context(|| format!("read expert {expert}, row {index}"))?;
            ensure!(row[..logical].iter().all(|v| v.is_finite()),
                "non-finite or missing value at expert {expert}, row {index}");
            row[logical..].fill(0.0);
            let bytes = encode_rows(layout, &row)?;
            ensure!(bytes.len() as u64 == layout.row_bytes(), "encoded row size mismatch");
            writer.write_all(&bytes)?;
        }
    }
    Ok(layout.tensor_bytes())
}

/// Encode a bounded batch of physical rows, with explicit zero padding.
/// Batch processing amortizes HQ44 parallel scheduling without staging a tensor.
pub fn encode_rows(layout: &MatrixLayout, rows: &[f32]) -> Result<Vec<u8>> {
    encode_rows_with_importance(layout, rows, &[])
}

/// As `encode_rows`, with one HQ44 importance per physical column (empty: unit).
/// Every row of the batch must belong to the same expert.
pub fn encode_rows_with_importance(layout: &MatrixLayout, rows: &[f32], importance: &[f32]) -> Result<Vec<u8>> {
    ensure!(importance.is_empty() || layout.storage() == Storage::Hq44, "importance only applies to HQ44");
    let physical = layout.physical_cols() as usize;
    let logical = layout.logical_cols() as usize;
    ensure!(!rows.is_empty() && rows.len() % physical == 0, "incomplete physical rows");
    ensure!(rows.iter().all(|v| v.is_finite()), "non-finite row value");
    ensure!(rows.chunks_exact(physical).all(|r| r[logical..].iter().all(|v| *v == 0.0)),
        "physical row padding must be zero");
    let count = rows.len() / physical;
    let size = layout.row_bytes().checked_mul(count as u64).context("batch byte overflow")?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(usize::try_from(size)?).context("encoded batch allocation failed")?;
    match layout.storage() {
        Storage::Fp32 => {
            for value in rows { bytes.extend_from_slice(&value.to_le_bytes()); }
        }
        Storage::Bf16 => {
            for value in rows {
                let converted = bf16::from_f32(*value);
                ensure!(converted.is_finite(), "BF16 overflow");
                bytes.extend_from_slice(&converted.to_le_bytes());
            }
        }
        Storage::Fp16 => {
            for value in rows {
                let converted = f16::from_f32(*value);
                ensure!(converted.is_finite(), "FP16 overflow");
                bytes.extend_from_slice(&converted.to_le_bytes());
            }
        }
        Storage::Hq8 => {
            bytes = crate::hqs::quantize_hq8k(rows);
        }
        Storage::Hq44 => {
            bytes = crate::hqs::v4::quantize(rows, physical, importance)?;
            for block in bytes.chunks_exact(crate::hqs::v4::BLOCK_BYTES) {
                for offset in [0, 2] {
                    ensure!(f16::from_le_bytes([block[offset], block[offset + 1]]).is_finite(),
                        "HQ44 base overflow");
                }
            }
        }
    }
    ensure!(bytes.len() as u64 == size, "encoded batch size mismatch");
    Ok(bytes)
}

/// Compatibility alias retained for early fixture clients.
pub use write_candidate_payload as write_payload;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn preserves_bf16_bits_expert_order_halves_and_padding() {
        let layout = MatrixLayout::new(1, Storage::Bf16, 4, 3, 3).unwrap();
        let mut bytes = Vec::new();
        let written = write_candidate_payload(&mut bytes, &layout, |expert, row, values| {
            for (col, v) in values.iter_mut().enumerate() {
                *v = bf16::from_bits(0x3f00 + (expert * 32 + row * 4 + col as u32) as u16).to_f32();
            }
            Ok(())
        }).unwrap();
        assert_eq!(written, bytes.len() as u64);
        for expert in 0..3 {
            let range = layout.expert_range(0, expert, written).unwrap();
            for row in 0..4 {
                let start = range.start as usize + row * layout.row_bytes() as usize;
                for col in 0..3 {
                    let at = start + col * 2;
                    assert_eq!(u16::from_le_bytes([bytes[at], bytes[at+1]]),
                        0x3f00 + (expert * 32 + row as u32 * 4 + col as u32) as u16);
                }
                assert!(bytes[start+6..start+layout.row_bytes() as usize].iter().all(|b| *b == 0));
            }
        }
    }

    #[test]
    fn hq44_rows_do_not_mix_at_704_and_2112_columns() {
        for width in [704, 2112] {
            let layout = MatrixLayout::new(2, Storage::Hq44, 2, width, 2).unwrap();
            let mut bytes = Vec::new();
            write_candidate_payload(&mut bytes, &layout, |expert, row, values| {
                values.fill(-0.25 * (1 + expert * 2 + row) as f32);
                Ok(())
            }).unwrap();
            let back = crate::hqs::v4::dequantize(&bytes, 4 * layout.physical_cols() as usize).unwrap();
            for (index, row) in back.chunks_exact(layout.physical_cols() as usize).enumerate() {
                let expected = -0.25 * (index + 1) as f32;
                assert!(row[..width as usize].iter().all(|v| (*v - expected).abs() < 0.003));
                assert!(row[width as usize..].iter().all(|v| *v == 0.0));
            }
        }
    }

    #[test]
    fn rejects_bad_ranges_and_layout_overflows() {
        let layout = MatrixLayout::new(2, Storage::Bf16, 2816, 704, 128).unwrap();
        let total = 17 + layout.tensor_bytes();
        assert_eq!(layout.expert_range(17, 0, total).unwrap().start, 17);
        assert_eq!(layout.expert_range(17, 127, total).unwrap().end, total);
        assert!(layout.expert_range(17, 128, total).is_err());
        assert!(layout.expert_range(17, 127, total - 1).is_err());
        assert!(layout.expert_range(u64::MAX, 0, u64::MAX).is_err());
        assert!(MatrixLayout::new(2, Storage::Fp32, u32::MAX, 256, u32::MAX).is_err());
        assert!(MatrixLayout::new(2, Storage::Bf16, 1, u32::MAX, 1).is_err());
        assert!(MatrixLayout::new(1, Storage::Bf16, 3, 256, 1).is_err());
        assert!(MatrixLayout::new(3, Storage::Bf16, 2, 256, 2).is_err());
        assert!(MatrixLayout::new(2, Storage::Bf16, 0, 256, 2).is_err());
    }

    #[test]
    fn rejects_non_finite_incomplete_and_unrepresentable_rows() {
        for storage in [Storage::Fp16, Storage::Bf16, Storage::Fp32, Storage::Hq44] {
            let layout = MatrixLayout::new(2, storage, 1, 3, 1).unwrap();
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                assert!(write_candidate_payload(&mut Vec::new(), &layout, |_, _, row| {
                    row.fill(bad); Ok(())
                }).is_err());
            }
            assert!(write_candidate_payload(&mut Vec::new(), &layout, |_, _, row| {
                row[0] = 1.0; Ok(())
            }).is_err());
        }
        for storage in [Storage::Fp16, Storage::Bf16, Storage::Hq44] {
            let layout = MatrixLayout::new(2, storage, 1, 256, 1).unwrap();
            assert!(write_candidate_payload(&mut Vec::new(), &layout, |_, _, row| {
                row.fill(f32::MAX); Ok(())
            }).is_err());
        }
    }

    #[test]
    fn propagates_source_and_output_errors() {
        let layout = MatrixLayout::new(2, Storage::Fp32, 1, 3, 1).unwrap();
        assert!(write_candidate_payload(&mut Vec::new(), &layout, |_, _, _| {
            anyhow::bail!("missing source row")
        }).is_err());
        struct BrokenWriter;
        impl Write for BrokenWriter {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> { Err(io::Error::other("disk full")) }
            fn flush(&mut self) -> io::Result<()> { Ok(()) }
        }
        assert!(write_candidate_payload(&mut BrokenWriter, &layout, |_, _, row| {
            row.fill(1.0); Ok(())
        }).is_err());
    }
}
