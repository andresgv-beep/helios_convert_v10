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
//! Por defecto todo va en FP16, tal como viene: es lo que hace falta para comparar el motor con
//! la referencia sin el ruido de la cuantización. Con `Levels` se elige el escalón de cada grupo
//! de matrices por separado (embedding, cabeza, atención, capa lineal, MLP y expertos), para
//! poder medir qué aguanta cada uno. Normas, vectores, la convolución, las puertas `lin.a` y
//! `lin.b`, el enrutador y la puerta del experto compartido se quedan siempre en FP16. Se dejan
//! fuera la torre visual (`model.visual.*`) y el módulo de predicción múltiple (`mtp.*`).
use crate::file::{open, verify_tensors, Builder};
use crate::format::align_up;
use crate::hints::binary::build_execution_hints_binary;
use crate::metadata::{kind, step, Component, Metadata, Property, Step, Tensor, Value};
use crate::SafetensorReader;
use anyhow::{bail, ensure, Context, Result};
use half::{bf16, f16};
use rayon::prelude::*;
use serde_json::json;
use std::io::Write;
use std::path::Path;

pub struct Options<'a> {
    pub model: &'a Path,
    pub out: &'a Path,
    pub commit: &'a str,
    pub levels: Levels,
}

/// El escalón de cada grupo de matrices. Los nombres son los dtype del HNF.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Levels {
    pub emb: &'static str, pub head: &'static str, pub attn: &'static str,
    pub lin: &'static str, pub mlp: &'static str, pub experts: &'static str,
}

impl Levels {
    pub fn fp16() -> Self { Levels { emb: "fp16", head: "fp16", attn: "fp16", lin: "fp16", mlp: "fp16", experts: "fp16" } }

    /// `grupo=escalón` separados por comas, sobre una base en FP16. Grupos: `emb`, `head`,
    /// `attn`, `lin`, `mlp`, `experts` y `todo`. Escalones: `fp16`, `hq8k`, `hq52`, `hq44`.
    /// Los expertos solo admiten `fp16` o `hq44`, que es lo que lee el MoE del motor.
    pub fn parse(spec: &str) -> Result<Self> {
        let mut levels = Levels::fp16();
        for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (group, level) = part.split_once('=').with_context(|| format!("«{part}»: se espera grupo=escalón"))?;
            let level = match level {
                "fp16" => "fp16", "hq8k" | "hq8" => "hq8k", "hq52k" | "hq52" => "hq52k",
                "hq44" | "hq44k" => crate::hqs::v4::DTYPE,
                other => bail!("escalón desconocido: {other} (fp16, hq8k, hq52 o hq44)"),
            };
            match group {
                "emb" => levels.emb = level, "head" => levels.head = level, "attn" => levels.attn = level,
                "lin" => levels.lin = level, "mlp" => levels.mlp = level, "experts" => levels.experts = level,
                "todo" => {
                    levels = Levels { emb: level, head: level, attn: level, lin: level, mlp: level, experts: levels.experts };
                }
                other => bail!("grupo desconocido: {other} (emb, head, attn, lin, mlp, experts o todo)"),
            }
        }
        ensure!(levels.experts == "fp16" || levels.experts == crate::hqs::v4::DTYPE,
            "los expertos solo van en fp16 o hq44: el MoE del motor no lee otro escalón");
        Ok(levels)
    }

    pub fn describe(&self) -> String {
        format!("emb={},head={},attn={},lin={},mlp={},experts={}", self.emb, self.head, self.attn, self.lin, self.mlp, self.experts)
    }

    /// El escalón de un tensor por su nombre canónico (sin el prefijo `text.`).
    fn of(&self, canonical: &str, shape: &[usize]) -> Result<&'static str> {
        let part = canonical.rsplit_once('.').map_or(canonical, |(head, _)| head);
        let part = part.split_once('.').filter(|(first, _)| first.starts_with("layer")).map_or(part, |(_, rest)| rest);
        let level = match part {
            "token_embedding" => self.emb,
            "lm_head" => self.head,
            "attn.q_proj" | "attn.k_proj" | "attn.v_proj" | "attn.o_proj" => self.attn,
            "lin.qkv" | "lin.z" | "lin.out" => self.lin,
            "mlp.gate" | "mlp.up" | "mlp.down" => self.mlp,
            "moe.experts.gate_up" | "moe.experts.down" => self.experts,
            _ => "fp16",
        };
        ensure!(level == "fp16" || (shape.len() >= 2 && shape.last().map_or(false, |&c| c % 256 == 0)),
            "{canonical}: {level} pide filas de un múltiplo de 256 columnas y tiene {:?}", shape);
        Ok(level)
    }
}

fn stored_bytes(dtype: &str, elements: u64) -> u64 {
    match dtype {
        "hq8k" => crate::hqs::hq8k_size(elements as usize) as u64,
        "hq52k" => crate::hqs::hq52k_size(elements as usize) as u64,
        "hq44k_g16" => elements / 256 * 152,
        _ => elements * 2,
    }
}

/// Filas por tanda al cuantizar: acota la RAM con el embedding y con los expertos.
const ROWS_PER_BATCH: usize = 8192;

const TENSOR_ALIGN: u64 = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Change { None, PlusOne }

#[derive(Debug, PartialEq)]
pub struct Mapped { pub canonical: String, pub layer: i32, pub plus_one: bool }

/// Nombre canónico de un tensor del checkpoint, o `None` si no es del modelo de texto.
pub fn map_tensor(name: &str) -> Option<Mapped> {
    let global = |canonical: &str, plus_one| Some(Mapped { canonical: canonical.into(), layer: -1, plus_one });
    // La cabeza propia, en los modelos que no la comparten con el embedding.
    if name == "lm_head.weight" { return global("lm_head.weight", false); }
    let rest = name.strip_prefix("model.language_model.")?;
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
        // MoE. El experto compartido es una MLP densa y ocupa los nombres de la MLP; ojo,
        // que en el checkpoint `mlp.gate` es el enrutador y no la puerta de una MLP.
        "mlp.gate.weight" => ("moe.router.weight", false),
        "mlp.experts.gate_up_proj" => ("moe.experts.gate_up.weight", false),
        "mlp.experts.down_proj" => ("moe.experts.down.weight", false),
        "mlp.shared_expert.gate_proj.weight" => ("mlp.gate.weight", false),
        "mlp.shared_expert.up_proj.weight" => ("mlp.up.weight", false),
        "mlp.shared_expert.down_proj.weight" => ("mlp.down.weight", false),
        "mlp.shared_expert_gate.weight" => ("moe.shared_gate.weight", false),
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
    name.starts_with("model.visual.") || name.starts_with("mtp.")
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
/// más el embedding y la norma final. Con MoE cada capa cambia su MLP de 3 tensores por 7
/// (enrutador, dos de expertos, tres del experto compartido y su puerta). La cabeza cuenta
/// si no se comparte con el embedding.
pub fn expected_tensors(letters: &str, moe: bool, tied: bool) -> usize {
    let per_layer_extra = if moe { 4 } else { 0 };
    2 + usize::from(!tied) + letters.chars().map(|c| (if c == 'l' { 14 } else { 11 }) + per_layer_extra).sum::<usize>()
}

fn source_f32(raw: &[u8], dtype: &str, i: usize) -> f32 {
    match dtype {
        "F32" => f32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap()),
        "BF16" => bf16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]).to_f32(),
        _ => f16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]).to_f32(),
    }
}

struct Planned { canonical: String, source: String, layer: i32, shape: Vec<usize>, change: Change, dtype: &'static str, bytes: u64 }

/// Escribe el tensor en FP16. Devuelve cuántos valores no caben exactos en FP16, para dejarlo
/// dicho en la procedencia: BF16 tiene más exponente y menos mantisa que FP16.
fn emit(w: &mut dyn Write, t: &Planned, raw: &[u8], dtype: &str) -> Result<u64> {
    let elements: usize = t.shape.iter().product();
    let width = if dtype == "F32" { 4 } else { 2 };
    ensure!(raw.len() == elements * width, "{}: bytes de origen que no cuadran", t.canonical);
    if t.dtype != "fp16" {
        ensure!(t.change == Change::None, "{}: una norma no se cuantiza", t.canonical);
        let row = *t.shape.last().context("forma vacía")?;
        let batch = row * ROWS_PER_BATCH;
        for start in (0..elements).step_by(batch) {
            // Pasar a f32 en paralelo: hecho en serie era lo que frenaba la conversión del
            // 35B, con los cuantizadores (que ya van en paralelo) esperando.
            let values: Vec<f32> = (start..elements.min(start + batch)).into_par_iter().map(|i| source_f32(raw, dtype, i)).collect();
            ensure!(values.par_iter().all(|v| v.is_finite()), "{}: valores no finitos", t.canonical);
            match t.dtype {
                "hq8k" => w.write_all(&crate::hqs::quantize_hq8k(&values))?,
                "hq52k" => w.write_all(&crate::hqs::quantize_hq52k(&values))?,
                // Sin calibrar: importancia 1 en todas las columnas.
                _ => w.write_all(&crate::hqs::v4::quantize(&values, row, &[])?)?,
            }
        }
        return Ok(0);
    }
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
        // En un MoE la MLP densa de la capa es el experto compartido.
        "intermediate_size": field(if text["num_experts"].is_null() { "intermediate_size" } else { "shared_expert_intermediate_size" })?,
        "vocab_size": vocab,
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
    ensure!(root["model_type"] == "qwen3_5" || root["model_type"] == "qwen3_5_moe",
        "requiere un checkpoint de la familia Qwen3.5 (model_type qwen3_5 o qwen3_5_moe)");
    let text = &root["text_config"];
    let moe = !text["num_experts"].is_null();
    let tied = text["tie_word_embeddings"].as_bool().unwrap_or(false);
    let letters = layer_letters(text)?;
    ensure!(letters.len() as u64 == text["num_hidden_layers"].as_u64().unwrap_or(0), "layer_types no cuadra con num_hidden_layers");
    let reader = SafetensorReader::from_folder(o.model)?;

    let mut plan = Vec::new();
    for (name, info) in reader.iter_tensors() {
        if tied && name == "lm_head.weight" { continue; }
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
        let dtype = o.levels.of(&m.canonical, &shape)?;
        plan.push(Planned { canonical: format!("text.{}", m.canonical), source: name.to_string(), layer: m.layer,
            shape, change: if m.plus_one { Change::PlusOne } else { Change::None }, dtype, bytes: stored_bytes(dtype, elements) });
    }
    plan.sort_by(|a, b| (a.layer, &a.canonical).cmp(&(b.layer, &b.canonical)));
    let expected = expected_tensors(&letters, moe, tied);
    ensure!(plan.len() == expected, "{} tensores de texto y se esperaban {expected}", plan.len());
    for (i, letter) in letters.chars().enumerate() {
        let has = |suffix: &str| plan.iter().any(|t| t.canonical == format!("text.layer{i}.{suffix}"));
        let ok = if letter == 'l' { has("lin.qkv.weight") && !has("attn.q_proj.weight") } else { has("attn.q_proj.weight") && !has("lin.qkv.weight") };
        ensure!(ok, "la capa {i} no trae los tensores de su tipo ({letter})");
        ensure!(has("moe.router.weight") == moe && has("moe.experts.gate_up.weight") == moe && has("mlp.down.weight"),
            "la capa {i} no trae la MLP que dice la configuración");
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
        md.tensors.push(Tensor { name: t.canonical.clone(), component: 1, layer: t.layer, dtype: t.dtype.into(),
            shape: t.shape.iter().map(|&d| d as u64).collect(), logical_last: *t.shape.last().unwrap() as u64,
            offset: text_at + layout[i], size: t.bytes,
            experts: if t.canonical.contains(".moe.experts.") { u32::try_from(t.shape[0])? } else { 0 }, hash: hashes[i].0,
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
    let mut properties = vec![
        ("layer_types", Value::Str(letters.clone())),
        ("linear_num_key_heads", integer("linear_num_key_heads")?),
        ("linear_num_value_heads", integer("linear_num_value_heads")?),
        ("linear_key_head_dim", integer("linear_key_head_dim")?),
        ("linear_value_head_dim", integer("linear_value_head_dim")?),
        ("linear_conv_kernel", integer("linear_conv_kernel_dim")?),
        ("attn_output_gate", Value::Bool(text["attn_output_gate"].as_bool().unwrap_or(false))),
    ];
    if moe {
        properties.push(("moe_experts", integer("num_experts")?));
        properties.push(("moe_top_k", integer("num_experts_per_tok")?));
        properties.push(("moe_intermediate", integer("moe_intermediate_size")?));
    }
    for (key, value) in properties {
        md.properties.push((3, Property { key: key.into(), value }));
    }
    for i in 0..md.components.len() { md.components[i].composite = md.composite(&md.components[i]); }

    let identity = md.identity();
    md.provenance.push(Step { kind: step::CONVERT, state: step::COMPLETE, tool: "hnf10 convert".into(),
        version: env!("CARGO_PKG_VERSION").into(), commit: o.commit.into(), input: [0; 16], output: identity,
        calibration: String::new(), touched: vec![], params: vec![
            Property { key: "family".into(), value: Value::Str("qwen3_5".into()) },
            Property { key: "profile".into(), value: Value::Str(o.levels.describe()) },
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
    fn the_moe_router_is_not_mistaken_for_a_dense_gate() {
        let at = |part: &str| map_tensor(&format!("model.language_model.layers.2.mlp.{part}")).unwrap().canonical;
        assert_eq!(at("gate.weight"), "layer2.moe.router.weight");
        assert_eq!(at("gate_proj.weight"), "layer2.mlp.gate.weight");
        assert_eq!(at("shared_expert.gate_proj.weight"), "layer2.mlp.gate.weight");
        assert_eq!(at("shared_expert_gate.weight"), "layer2.moe.shared_gate.weight");
        assert_eq!(at("experts.gate_up_proj"), "layer2.moe.experts.gate_up.weight");
        assert_eq!(at("experts.down_proj"), "layer2.moe.experts.down.weight");
        assert_eq!(map_tensor("lm_head.weight").unwrap().canonical, "lm_head.weight");
    }

    #[test]
    fn each_group_takes_its_own_level_and_small_tensors_stay_in_fp16() {
        let levels = Levels::parse("todo=hq8k,mlp=hq44,experts=hq44,head=hq52").unwrap();
        let of = |name: &str, shape: &[usize]| levels.of(name, shape).unwrap();
        assert_eq!(of("token_embedding.weight", &[1000, 1024]), "hq8k");
        assert_eq!(of("lm_head.weight", &[1000, 1024]), "hq52k");
        assert_eq!(of("layer3.attn.q_proj.weight", &[4096, 1024]), "hq8k");
        assert_eq!(of("layer0.lin.qkv.weight", &[6144, 1024]), "hq8k");
        assert_eq!(of("layer0.mlp.down.weight", &[1024, 3584]), crate::hqs::v4::DTYPE);
        assert_eq!(of("layer0.moe.experts.down.weight", &[256, 2048, 512]), crate::hqs::v4::DTYPE);
        for name in ["layer0.lin.a.weight", "layer0.lin.b.weight", "layer0.lin.conv.weight", "layer0.lin.a_log",
            "layer0.lin.norm.weight", "layer0.moe.router.weight", "layer0.moe.shared_gate.weight", "final_norm.weight"] {
            assert_eq!(of(name, &[16, 1024]), "fp16", "{name}");
        }
        assert_eq!(Levels::parse("").unwrap(), Levels::fp16());
        // «todo» no arrastra a los expertos: solo admiten fp16 o hq44.
        assert_eq!(Levels::parse("todo=hq8k").unwrap().experts, "fp16");
        assert!(Levels::parse("experts=hq8k").is_err());
        assert!(Levels::parse("mlp=q4").is_err() && Levels::parse("cosa=hq8k").is_err());
        assert!(levels.of("layer0.mlp.down.weight", &[1024, 1000]).is_err());
    }

    #[test]
    fn layer_types_and_the_tensor_count() {
        let text = json!({"layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]});
        let letters = layer_letters(&text).unwrap();
        assert_eq!(letters, "llla");
        assert_eq!(expected_tensors(&letters, false, true), 2 + 3 * 14 + 11);
        assert_eq!(expected_tensors(&letters, true, false), 3 + 3 * 18 + 15);
        assert!(layer_letters(&json!({"layer_types": ["sliding_attention"]})).is_err());
    }

    #[test]
    fn plus_one_is_applied_when_writing_and_inexact_values_are_counted() {
        let t = Planned { canonical: "text.final_norm.weight".into(), source: String::new(), layer: -1, shape: vec![2], dtype: "fp16",
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
