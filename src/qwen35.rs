//! Conversión directa de la familia Qwen3.5 densa (0.8B, 2B, 4B, 9B, 27B) de safetensors a HNFv10.
//!
//! La familia alterna dos tipos de capa: tres de cada cuatro son Gated DeltaNet (atención lineal
//! con un estado recurrente en lugar de caché) y la cuarta es atención con puerta de salida. La
//! MLP es la SwiGLU de siempre. El estudio está en Hector-engine,
//! `informes/estudio-qwen3.6-35b-2026-11-16`.
//!
//! Nombres canónicos. Lo que el motor ya conoce se llama como en Qwen3 (`attn.q_proj`,
//! `ln_attn_in`, `mlp.gate`…). Lo nuevo va bajo `lin.`:
//!
//! | checkpoint (`linear_attn.`) | canónico | |
//! |---|---|---|
//! | `in_proj_qkv.weight` | `lin.qkv.weight` | consulta, clave y valor, antes de la convolución |
//! | `in_proj_z.weight` | `lin.z.weight` | la puerta de la norma de salida |
//! | `in_proj_a.weight`, `in_proj_b.weight` | `lin.a.weight`, `lin.b.weight` | olvido y escritura, una fila por cabeza |
//! | `conv1d.weight` `[C, 1, K]` | `lin.conv.weight` `[C, K]` | convolución causal canal a canal |
//! | `A_log`, `dt_bias` | `lin.a_log`, `lin.dt_bias` | |
//! | `norm.weight` | `lin.norm.weight` | norma por cabeza de valor |
//! | `out_proj.weight` | `lin.out.weight` | |
//!
//! **Las normas de la familia van centradas en cero**: la referencia multiplica por `1 + peso`.
//! Aquí se guarda ya `1 + peso` en las normas de capa, la final y las de consulta y clave, para
//! que el motor las aplique como cualquier otra RMSNorm. La norma `lin.norm` no: esa usa el peso
//! tal cual. Convertir sin esto da un modelo que arranca y no dice nada con sentido.
//!
//! En las capas de atención `q_proj` tiene el doble de filas: por cabeza, la primera mitad es
//! la consulta y la segunda una puerta. Se guarda entera y el motor la parte.
//!
//! Los datos propios de la familia viajan como propiedades de la configuración (`layer_types`,
//! `linear_*`, `attn_output_gate`): la configuración binaria no los tiene y no se toca.
//!
//! De momento un solo perfil, `fp16`: todo tal como viene, sin cuantizar. Es el que hace falta
//! para comparar el motor con la referencia sin el ruido de la cuantización. Se dejan fuera la
//! torre visual (`model.visual.*`) y el módulo de predicción múltiple (`mtp.*`).
use crate::file::{open, verify_tensors, Builder};
use crate::format::align_up;
use crate::hints::binary::build_execution_hints_binary;
use crate::metadata::{kind, step, Component, Metadata, Property, Step, Tensor, Value};
use crate::SafetensorReader;
use anyhow::{bail, ensure, Context, Result};
use half::{bf16, f16};
use serde_json::json;
use std::io::Write;
use std::path::Path;

pub struct Options<'a> {
    pub model: &'a Path,
    pub out: &'a Path,
    pub commit: &'a str,
}

const TENSOR_ALIGN: u64 = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Change { None, PlusOne }

#[derive(Debug, PartialEq)]
pub struct Mapped { pub canonical: String, pub layer: i32, pub plus_one: bool }

/// Nombre canónico de un tensor del checkpoint, o `None` si no es del modelo de texto.
pub fn map_tensor(name: &str) -> Option<Mapped> {
    let rest = name.strip_prefix("model.language_model.")?;
    let global = |canonical: &str, plus_one| Some(Mapped { canonical: canonical.into(), layer: -1, plus_one });
    match rest {
        "embed_tokens.weight" => return global("token_embedding.weight", false),
        "norm.weight" => return global("final_norm.weight", true),
        _ => {}
    }
    let rest = rest.strip_prefix("layers.")?;
    let (index, part) = rest.split_once('.')?;
    let layer: i32 = index.parse().ok()?;
    let (canonical, plus_one) = match part {
        "input_layernorm.weight" => ("ln_attn_in.weight", true),
        "post_attention_layernorm.weight" => ("ln_attn_out.weight", true),
        "mlp.gate_proj.weight" => ("mlp.gate.weight", false),
        "mlp.up_proj.weight" => ("mlp.up.weight", false),
        "mlp.down_proj.weight" => ("mlp.down.weight", false),
        "self_attn.q_proj.weight" => ("attn.q_proj.weight", false),
        "self_attn.k_proj.weight" => ("attn.k_proj.weight", false),
        "self_attn.v_proj.weight" => ("attn.v_proj.weight", false),
        "self_attn.o_proj.weight" => ("attn.o_proj.weight", false),
        "self_attn.q_norm.weight" => ("attn.q_norm.weight", true),
        "self_attn.k_norm.weight" => ("attn.k_norm.weight", true),
        "linear_attn.in_proj_qkv.weight" => ("lin.qkv.weight", false),
        "linear_attn.in_proj_z.weight" => ("lin.z.weight", false),
        "linear_attn.in_proj_a.weight" => ("lin.a.weight", false),
        "linear_attn.in_proj_b.weight" => ("lin.b.weight", false),
        "linear_attn.conv1d.weight" => ("lin.conv.weight", false),
        "linear_attn.A_log" => ("lin.a_log", false),
        "linear_attn.dt_bias" => ("lin.dt_bias", false),
        "linear_attn.norm.weight" => ("lin.norm.weight", false),
        "linear_attn.out_proj.weight" => ("lin.out.weight", false),
        _ => return None,
    };
    Some(Mapped { canonical: format!("layer{layer}.{canonical}"), layer, plus_one })
}

/// Tensores que se dejan fuera a propósito. Cualquier otro sin nombre canónico es un error:
/// mejor parar que convertir un modelo al que le falta una pieza.
fn skipped(name: &str) -> bool {
    name.starts_with("model.visual.") || name.starts_with("mtp.") || name == "lm_head.weight"
}

/// Los tipos de capa en una letra cada uno: `l` lineal (DeltaNet), `a` atención.
pub fn layer_letters(text: &serde_json::Value) -> Result<String> {
    let types = text["layer_types"].as_array().context("falta layer_types")?;
    types.iter().map(|t| match t.as_str() {
        Some("linear_attention") => Ok('l'),
        Some("full_attention") => Ok('a'),
        other => bail!("tipo de capa desconocido: {other:?}"),
    }).collect()
}

/// Cuántos tensores de texto tiene que haber: 14 por capa lineal, 11 por capa de atención,
/// más el embedding y la norma final.
pub fn expected_tensors(letters: &str) -> usize {
    2 + letters.chars().map(|c| if c == 'l' { 14 } else { 11 }).sum::<usize>()
}

fn source_f32(raw: &[u8], dtype: &str, i: usize) -> f32 {
    match dtype {
        "F32" => f32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap()),
        "BF16" => bf16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]).to_f32(),
        _ => f16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]).to_f32(),
    }
}

struct Planned { canonical: String, source: String, layer: i32, shape: Vec<usize>, change: Change, bytes: u64 }

/// Escribe el tensor en FP16. Devuelve cuántos valores no caben exactos en FP16, para dejarlo
/// dicho en la procedencia: BF16 tiene más exponente y menos mantisa que FP16.
fn emit(w: &mut dyn Write, t: &Planned, raw: &[u8], dtype: &str) -> Result<u64> {
    let elements: usize = t.shape.iter().product();
    let width = if dtype == "F32" { 4 } else { 2 };
    ensure!(raw.len() == elements * width, "{}: bytes de origen que no cuadran", t.canonical);
    let mut inexact = 0u64;
    let mut bytes = Vec::with_capacity(elements.min(1 << 20) * 2);
    for i in 0..elements {
        let mut v = source_f32(raw, dtype, i);
        ensure!(v.is_finite(), "{}: valores no finitos", t.canonical);
        if t.change == Change::PlusOne { v += 1.0; }
        let y = f16::from_f32(v);
        ensure!(y.is_finite(), "{}: valor fuera de FP16", t.canonical);
        if y.to_f32() != v { inexact += 1; }
        bytes.extend_from_slice(&y.to_le_bytes());
        if bytes.len() >= 1 << 21 { w.write_all(&bytes)?; bytes.clear(); }
    }
    w.write_all(&bytes)?;
    Ok(inexact)
}

struct Hashing<'a> { inner: &'a mut dyn Write, state: xxhash_rust::xxh3::Xxh3, written: u64 }
impl Write for Hashing<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?; self.state.update(&buf[..n]); self.written += n as u64; Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> { self.inner.flush() }
}

/// La configuración de texto que entiende el motor. `arch` es `qwen2` porque el camino general
/// es el suyo; lo que distingue a la familia va en las propiedades.
fn text_hints(text: &serde_json::Value, vocab: usize) -> Result<serde_json::Value> {
    let field = |key: &str| text[key].as_u64().with_context(|| format!("falta {key}"));
    let rope = &text["rope_parameters"];
    Ok(json!({
        "arch": "qwen2", "dtype": "fp16",
        "num_hidden_layers": field("num_hidden_layers")?, "hidden_size": field("hidden_size")?,
        "intermediate_size": field("intermediate_size")?, "vocab_size": vocab,
        "num_attention_heads": field("num_attention_heads")?, "num_key_value_heads": field("num_key_value_heads")?,
        "head_dim": field("head_dim")?, "attention_type": "gqa", "attention_bias": false,
        "qkv_layout": "separate", "use_qk_norm": true, "parallel_attention": false, "kv_layout": "BHSD",
        "mlp_type": "swiglu", "mlp_activation": "silu", "mlp_bias": false,
        "norm_type": "rmsnorm", "norm_bias": false, "rms_norm_eps": text["rms_norm_eps"].as_f64().context("falta rms_norm_eps")?,
        "pre_norm": true, "final_norm": true,
        "rope_type": "default", "rope_theta": rope["rope_theta"].as_f64().context("falta rope_theta")?,
        "partial_rotary_factor": rope["partial_rotary_factor"].as_f64().unwrap_or(1.0),
        "rope_dim": field("head_dim")?, "rope_partial": rope["partial_rotary_factor"].as_f64().unwrap_or(1.0) < 1.0,
        "rope_interleaved": false,
        "tie_word_embeddings": text["tie_word_embeddings"].as_bool().unwrap_or(false),
        "embedding_bias": false, "lm_head_bias": false,
        "max_position_embeddings": field("max_position_embeddings")?,
        "moe_enabled": false,
        "supports_flash_attention": false, "supports_paged_attention": false, "supports_sdpa": false
    }))
}

pub fn convert(o: &Options) -> Result<[u8; 16]> {
    ensure!(!o.out.exists(), "{} ya existe", o.out.display());
    let root: serde_json::Value = serde_json::from_slice(&std::fs::read(o.model.join("config.json"))?)?;
    ensure!(root["model_type"] == "qwen3_5", "requiere un checkpoint Qwen3.5 denso (model_type qwen3_5)");
    let text = &root["text_config"];
    ensure!(text["num_experts"].is_null(), "es un MoE: esta conversión solo cubre los densos");
    ensure!(text["tie_word_embeddings"].as_bool().unwrap_or(false),
        "embeddings y cabeza sin compartir: falta decidir el formato de lm_head para este tamaño");
    let letters = layer_letters(text)?;
    ensure!(letters.len() as u64 == text["num_hidden_layers"].as_u64().unwrap_or(0), "layer_types no cuadra con num_hidden_layers");
    let reader = SafetensorReader::from_folder(o.model)?;

    let mut plan = Vec::new();
    for (name, info) in reader.iter_tensors() {
        let Some(m) = map_tensor(name) else {
            ensure!(skipped(name), "{name}: tensor que esta conversión no conoce");
            continue;
        };
        ensure!(["BF16", "F16", "F32"].contains(&info.dtype.as_str()), "{name}: dtype {} no admitido", info.dtype);
        // La convolución trae una dimensión de un canal que no dice nada: [C, 1, K] -> [C, K].
        let shape: Vec<usize> = if m.canonical.ends_with("lin.conv.weight") {
            ensure!(info.shape.len() == 3 && info.shape[1] == 1, "{name}: se esperaba [C, 1, K]");
            vec![info.shape[0], info.shape[2]]
        } else { info.shape.clone() };
        let elements: u64 = shape.iter().map(|&d| d as u64).product();
        plan.push(Planned { canonical: format!("text.{}", m.canonical), source: name.to_string(), layer: m.layer,
            shape, change: if m.plus_one { Change::PlusOne } else { Change::None }, bytes: elements * 2 });
    }
    plan.sort_by(|a, b| (a.layer, &a.canonical).cmp(&(b.layer, &b.canonical)));
    ensure!(plan.len() == expected_tensors(&letters), "{} tensores de texto y se esperaban {}", plan.len(), expected_tensors(&letters));
    for (i, letter) in letters.chars().enumerate() {
        let has = |suffix: &str| plan.iter().any(|t| t.canonical == format!("text.layer{i}.{suffix}"));
        let ok = if letter == 'l' { has("lin.qkv.weight") && !has("attn.q_proj.weight") } else { has("attn.q_proj.weight") && !has("lin.qkv.weight") };
        ensure!(ok, "la capa {i} no trae los tensores de su tipo ({letter})");
    }

    let embedding = plan.iter().find(|t| t.canonical == "text.token_embedding.weight").context("falta el embedding")?;
    let config = build_execution_hints_binary(&text_hints(text, embedding.shape[0])?)?;
    let tokenizer = crate::htf::build_htf(o.model)?;
    ensure!(tokenizer.len() > 64, "falta el tokenizer");
    let template = std::fs::read(o.model.join("chat_template.jinja")).ok();
    let base_id = crate::base_id::compute(o.model)?;

    let mut layout = Vec::with_capacity(plan.len());
    let mut end = 0u64;
    for t in &plan { let at = align_up(end, TENSOR_ALIGN); layout.push(at); end = at + t.bytes; }
    let text_size = end;

    let mut hashes = Vec::with_capacity(plan.len());
    let mut inexact = 0u64;
    let mut builder = Builder::create(o.out)?;
    let (text_at, _) = builder.payload(text_size, 4096, |w| {
        let mut written = 0u64;
        for (t, &at) in plan.iter().zip(&layout) {
            w.write_all(&vec![0u8; (at - written) as usize])?;
            let info = reader.tensor_info(&t.source).unwrap();
            let raw = reader.read_raw(&t.source)?;
            let mut hashing = Hashing { inner: &mut *w, state: xxhash_rust::xxh3::Xxh3::new(), written: 0 };
            inexact += emit(&mut hashing, t, raw, &info.dtype)?;
            ensure!(hashing.written == t.bytes, "{}: {} bytes escritos, {} previstos", t.canonical, hashing.written, t.bytes);
            hashes.push((hashing.state.digest128().to_le_bytes(), xxhash_rust::xxh3::xxh3_64(raw)));
            written = at + t.bytes;
        }
        Ok(())
    })?;
    let (tok_at, tok_hash) = builder.payload(tokenizer.len() as u64, 4096, |w| { w.write_all(&tokenizer)?; Ok(()) })?;
    let (cfg_at, cfg_hash) = builder.payload(config.len() as u64, 4096, |w| { w.write_all(&config)?; Ok(()) })?;
    let tpl = match &template {
        Some(bytes) => Some(builder.payload(bytes.len() as u64, 4096, |w| { w.write_all(bytes)?; Ok(()) })?),
        None => None,
    };

    let mut md = Metadata::default();
    let mut weight_bytes = 0;
    for (i, t) in plan.iter().enumerate() {
        md.tensors.push(Tensor { name: t.canonical.clone(), component: 1, layer: t.layer, dtype: "fp16".into(),
            shape: t.shape.iter().map(|&d| d as u64).collect(), logical_last: *t.shape.last().unwrap() as u64,
            offset: text_at + layout[i], size: t.bytes, experts: 0, hash: hashes[i].0,
            source: t.source.clone(), source_hash: hashes[i].1 });
        weight_bytes += t.bytes;
    }
    let n = md.tensors.len() as u32;
    let component = |id, kind, required, name: &str, offset, size, first, count, weight, hash| Component {
        id, kind, required, name: name.into(), payload_offset: offset, payload_size: size, alignment: 4096, weight_bytes: weight,
        memory: None, first_tensor: first, tensor_count: count, depends_on: None, composite: [0; 16], bytes_hash: hash };
    md.components.push(component(1, kind::TEXT, true, "texto", text_at, text_size, 0, n, weight_bytes, [0; 16]));
    md.components.push(component(2, kind::TOKENIZER, true, "tokenizer HTF3", tok_at, tokenizer.len() as u64, n, 0, 0, tok_hash));
    md.components.push(component(3, kind::CONFIG, true, "configuración TNIH", cfg_at, config.len() as u64, n, 0, 0, cfg_hash));
    if let (Some((at, hash)), Some(bytes)) = (tpl, &template) {
        md.components.push(component(4, kind::TEMPLATE, false, "chat_template.jinja", at, bytes.len() as u64, n, 0, 0, hash));
    }
    // Lo que distingue a la familia y la configuración binaria no trae.
    let integer = |key: &str| -> Result<Value> { Ok(Value::Int(text[key].as_i64().with_context(|| format!("falta {key}"))?)) };
    for (key, value) in [
        ("layer_types", Value::Str(letters.clone())),
        ("linear_num_key_heads", integer("linear_num_key_heads")?),
        ("linear_num_value_heads", integer("linear_num_value_heads")?),
        ("linear_key_head_dim", integer("linear_key_head_dim")?),
        ("linear_value_head_dim", integer("linear_value_head_dim")?),
        ("linear_conv_kernel", integer("linear_conv_kernel_dim")?),
        ("attn_output_gate", Value::Bool(text["attn_output_gate"].as_bool().unwrap_or(false))),
    ] {
        md.properties.push((3, Property { key: key.into(), value }));
    }
    for i in 0..md.components.len() { md.components[i].composite = md.composite(&md.components[i]); }

    let identity = md.identity();
    md.provenance.push(Step { kind: step::CONVERT, state: step::COMPLETE, tool: "hnf10 convert".into(),
        version: env!("CARGO_PKG_VERSION").into(), commit: o.commit.into(), input: [0; 16], output: identity,
        calibration: String::new(), touched: vec![], params: vec![
            Property { key: "family".into(), value: Value::Str("qwen3_5".into()) },
            Property { key: "profile".into(), value: Value::Str("fp16".into()) },
            Property { key: "norms".into(), value: Value::Str("stored as 1 + weight".into()) },
            Property { key: "fp16_inexact_values".into(), value: Value::Int(inexact as i64) },
            Property { key: "base_id".into(), value: Value::Str(base_id) },
            Property { key: "source_model".into(), value: Value::Str(o.model.display().to_string()) },
        ] });
    builder.finish(md)?;

    let opened = open(o.out)?;
    ensure!(opened.header.identity == identity, "identidad distinta al reabrir");
    let bad = verify_tensors(o.out, &opened.metadata)?;
    ensure!(bad.is_empty(), "tensores que no cuadran: {bad:?}");
    Ok(identity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_what_the_engine_already_knows_under_its_qwen3_names() {
        let m = map_tensor("model.language_model.layers.3.self_attn.q_proj.weight").unwrap();
        assert_eq!((m.canonical.as_str(), m.layer, m.plus_one), ("layer3.attn.q_proj.weight", 3, false));
        assert_eq!(map_tensor("model.language_model.layers.7.mlp.down_proj.weight").unwrap().canonical, "layer7.mlp.down.weight");
        assert_eq!(map_tensor("model.language_model.embed_tokens.weight").unwrap().canonical, "token_embedding.weight");
    }

    #[test]
    fn the_new_layer_goes_under_lin() {
        for (source, canonical) in [("in_proj_qkv.weight", "lin.qkv.weight"), ("in_proj_z.weight", "lin.z.weight"),
            ("in_proj_a.weight", "lin.a.weight"), ("in_proj_b.weight", "lin.b.weight"), ("conv1d.weight", "lin.conv.weight"),
            ("A_log", "lin.a_log"), ("dt_bias", "lin.dt_bias"), ("norm.weight", "lin.norm.weight"), ("out_proj.weight", "lin.out.weight")] {
            let m = map_tensor(&format!("model.language_model.layers.0.linear_attn.{source}")).unwrap();
            assert_eq!((m.canonical, m.plus_one), (format!("layer0.{canonical}"), false));
        }
    }

    #[test]
    fn zero_centred_norms_are_marked_and_the_gated_one_is_not() {
        for name in ["norm.weight", "layers.0.input_layernorm.weight", "layers.0.post_attention_layernorm.weight",
            "layers.3.self_attn.q_norm.weight", "layers.3.self_attn.k_norm.weight"] {
            assert!(map_tensor(&format!("model.language_model.{name}")).unwrap().plus_one, "{name}");
        }
        assert!(!map_tensor("model.language_model.layers.0.linear_attn.norm.weight").unwrap().plus_one);
    }

    #[test]
    fn vision_and_mtp_are_left_out_on_purpose_and_anything_else_is_an_error() {
        for name in ["model.visual.blocks.0.attn.qkv.weight", "mtp.fc.weight", "mtp.layers.0.mlp.up_proj.weight"] {
            assert!(map_tensor(name).is_none() && skipped(name), "{name}");
        }
        assert!(map_tensor("model.language_model.layers.0.algo_nuevo.weight").is_none());
        assert!(!skipped("model.language_model.layers.0.algo_nuevo.weight"));
    }

    #[test]
    fn layer_types_and_the_tensor_count() {
        let text = json!({"layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]});
        let letters = layer_letters(&text).unwrap();
        assert_eq!(letters, "llla");
        assert_eq!(expected_tensors(&letters), 2 + 3 * 14 + 11);
        assert!(layer_letters(&json!({"layer_types": ["sliding_attention"]})).is_err());
    }

    #[test]
    fn plus_one_is_applied_when_writing_and_inexact_values_are_counted() {
        let t = Planned { canonical: "text.final_norm.weight".into(), source: String::new(), layer: -1, shape: vec![2],
            change: Change::PlusOne, bytes: 4 };
        let raw: Vec<u8> = [0.5f32, -0.25].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut out = Vec::new();
        assert_eq!(emit(&mut out, &t, &raw, "F32").unwrap(), 0);
        let values: Vec<f32> = out.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect();
        assert_eq!(values, vec![1.5, 0.75]);
        let plain = Planned { change: Change::None, ..t };
        let raw: Vec<u8> = [0.1f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let fine = Planned { shape: vec![1], bytes: 2, ..plain };
        assert_eq!(emit(&mut Vec::new(), &fine, &raw, "F32").unwrap(), 1);
    }
}
