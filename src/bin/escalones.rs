//! Curva tamaño/precisión de los formatos HQS sobre pesos reales, para decidir un reparto
//! sin convertir un modelo entero por cada pregunta.
//!
//! Error = L2 relativo frente al original, `||q(w) - w|| / ||w||`, la misma definición que
//! docs/HQS_FORMATOS.md. Se cuantiza por filas completas, como el conversor, porque los
//! superbloques son de 256 columnas. De las tablas grandes se toma un trozo de filas.
//!
//! uso: escalones --model <checkpoint/> [--filas N] [--importancia <dir>]

use anyhow::{Context, Result};
use clap::Parser;
use half::bf16;
use helios_convert_v10::safetensor::SafetensorReader;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)] model: PathBuf,
    /// Filas que se toman de cada tensor (las tablas tienen 262.144).
    #[arg(long, default_value_t = 2048)] filas: usize,
    /// Directorio de importancia por columna, para medir qué aporta en HQ44.
    #[arg(long)] importancia: Option<PathBuf>,
}

fn a_f32(raw: &[u8], dtype: &str, i: usize) -> f32 {
    match dtype {
        "F32" => f32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap()),
        "BF16" => bf16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]).to_f32(),
        _ => half::f16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]).to_f32(),
    }
}

fn l2_relativo(original: &[f32], vuelta: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (a, b) in original.iter().zip(vuelta) {
        let d = (*b - *a) as f64;
        num += d * d;
        den += (*a as f64) * (*a as f64);
    }
    if den == 0.0 { 0.0 } else { (num / den).sqrt() }
}

fn main() -> Result<()> {
    let a = Args::parse();
    let reader = SafetensorReader::from_folder(&a.model)?;
    let p = "model.language_model.";
    let objetivos: Vec<(&str, String)> = vec![
        ("atención q", format!("{p}layers.0.self_attn.q_proj.weight")),
        ("atención o", format!("{p}layers.0.self_attn.o_proj.weight")),
        ("MLP gate", format!("{p}layers.0.mlp.gate_proj.weight")),
        ("MLP down", format!("{p}layers.0.mlp.down_proj.weight")),
        ("embedding", format!("{p}embed_tokens.weight")),
        ("tabla PLE", format!("{p}embed_tokens_per_layer.weight")),
    ];

    println!("{:<12} {:>9} {:>8}  {}", "tensor", "columnas", "filas", "error L2 relativo por formato");
    for (etiqueta, nombre) in &objetivos {
        let info = reader.tensor_info(nombre).with_context(|| format!("falta {nombre}"))?;
        let cols = *info.shape.last().context("forma vacía")?;
        let filas = a.filas.min(info.shape[0]);
        let dtype = info.dtype.clone();
        let raw = reader.read_raw(nombre)?;
        let original: Vec<f32> = (0..filas * cols).map(|i| a_f32(raw, &dtype, i)).collect();

        let mut linea = format!("{etiqueta:<12} {cols:>9} {filas:>8}  ");
        for formato in ["hq31k", "hq44k_g16", "hq52k", "hq62k", "hq8k"] {
            let (bytes, vuelta) = match formato {
                "hq31k" => (helios_convert_v10::hqs::hq31k_size(original.len()),
                            helios_convert_v10::hqs::dequantize_hq31k(
                                &helios_convert_v10::hqs::quantize_hq31k(&original, true), original.len())),
                "hq44k_g16" => (original.len() / 256 * 152,
                            helios_convert_v10::hqs::v4::dequantize(
                                &helios_convert_v10::hqs::v4::quantize(&original, cols, &[])?, original.len())?),
                "hq52k" => (helios_convert_v10::hqs::hq52k_size(original.len()),
                            helios_convert_v10::hqs::dequantize_hq52k(
                                &helios_convert_v10::hqs::quantize_hq52k(&original), original.len())),
                "hq62k" => (helios_convert_v10::hqs::hq62k_size(original.len()),
                            helios_convert_v10::hqs::dequantize_hq62k(
                                &helios_convert_v10::hqs::quantize_hq62k(&original, true), original.len())),
                _ => (helios_convert_v10::hqs::hq8k_size(original.len()),
                            helios_convert_v10::hqs::dequantize_hq8k(
                                &helios_convert_v10::hqs::quantize_hq8k(&original), original.len())),
            };
            let bits = bytes as f64 * 8.0 / original.len() as f64;
            linea += &format!("{formato} {:.2}b {:.3}%   ", bits, 100.0 * l2_relativo(&original, &vuelta));
        }
        println!("{linea}");

        // La importancia solo existe para los down, y es lo que distingue el HQ44 calibrado.
        if let (Some(dir), true) = (&a.importancia, nombre.ends_with("mlp.down_proj.weight")) {
            let canon = "text.layer0.mlp.down.weight";
            let bytes = std::fs::read(dir.join(format!("{canon}.f32")))?;
            let imp: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            anyhow::ensure!(imp.len() == cols, "la importancia trae {} valores y las columnas son {cols}", imp.len());
            let q = helios_convert_v10::hqs::v4::quantize(&original, cols, &imp)?;
            let vuelta = helios_convert_v10::hqs::v4::dequantize(&q, original.len())?;
            println!("{:<12} {:>9} {:>8}  hq44k_g16 calibrado 4.75b {:.3}%", "  (con imp)", cols, filas,
                     100.0 * l2_relativo(&original, &vuelta));
        }
    }
    Ok(())
}
