//! Conversión directa de Gemma 4 densa (E2B, E4B) de safetensors a HNFv10.
//!
//! Dos perfiles. `Q8`: toda matriz con columnas múltiplo de 256 (atención, MLP, embedding, PLE y
//! sus proyecciones) en HQ8K, para GPUs con VRAM de sobra; HQ8K sustituye a HQ62K, descartado el
//! 2026-11-02 (docs/HQS_FORMATOS.md). `Compacto`: además, MLP y tabla PLE en HQ44, las dos piezas
//! grandes; atención y embedding siguen en HQ8K, como en el 26B. Normas y escalares, en FP16. Con `vision`, la torre va en el mismo
//! fichero como componente VISION opcional (vision.rs). Cada tensor de texto va alineado a 4 KiB
//! y su xxh3-128 se calcula mientras se escribe, como en la conversión del MoE.
use crate::file::{open, verify_tensors, Builder};
use crate::format::align_up;
use crate::hints::binary::{append_vision, build_execution_hints_binary};
use crate::mapping::gemma4::Gemma4Mapper;
use crate::mapping::{ModelMapper, QuantHint};
use crate::metadata::{kind, step, Component, Metadata, Property, Step, Tensor, Value};
use crate::vision::Tower;
use crate::SafetensorReader;
use anyhow::{ensure, Context, Result};
use half::{bf16, f16};
use std::io::Write;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile { Q8, Compacto }

/// Formato de la tabla PLE, aparte del que le toque a la MLP por el perfil. El Q4_K_M de
/// Unsloth le da Q5_K y el v9.1 unos 5,6 bits, así que el HQ44 del compacto puede quedarle
/// corto (docs/HQS_FORMATOS.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ple { DelPerfil, Hq52, Hq8 }

pub struct Options<'a> {
    pub model: &'a Path,
    pub out: &'a Path,
    pub profile: Profile,
    pub ple: Ple,
    pub vision: bool,
    pub commit: &'a str,
    /// Directorio con la importancia por columna de los `mlp.down`, como el del v9.1
    /// (`data/hqs_v4/importance-down-e4b`). Sin él, el HQ44 va con importancia 1.
    pub importancia: Option<&'a Path>,
}

const TENSOR_ALIGN: u64 = 4096;
/// Filas por tanda al cuantizar: acota la RAM con la tabla PLE (262144 × 10752).
const ROWS_PER_BATCH: usize = 1024;

struct Planned { canonical: String, source: String, layer: i32, shape: Vec<usize>, dtype: &'static str, bytes: u64 }

fn source_f32(raw: &[u8], dtype: &str, i: usize) -> f32 {
    match dtype {
        "F32" => f32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap()),
        "BF16" => bf16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]).to_f32(),
        _ => f16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]).to_f32(),
    }
}

/// Cuantizada si la matriz admite superbloques de 256 por fila; si no, FP16. Las normas,
/// escalares y demás tensores que el mapper pide en FP16 se quedan en FP16, salvo el embedding,
/// que va en HQ8K como en el 26B. En el compacto, MLP y tabla PLE van en HQ44.
fn storage(profile: Profile, ple: Ple, canonical: &str, hint: QuantHint, shape: &[usize]) -> &'static str {
    let matrix = shape.len() >= 2 && shape.last().map_or(false, |&c| c % 256 == 0);
    let embedding = canonical == "text.token_embedding.weight";
    if !matrix || (hint == QuantHint::FP16 && !embedding) { return "fp16"; }
    let tabla_ple = canonical == "text.ple.token_embedding.weight";
    let large = canonical.contains(".mlp.") || tabla_ple;
    let del_perfil = if profile == Profile::Compacto && large { crate::hqs::v4::DTYPE } else { "hq8k" };
    if !tabla_ple { return del_perfil; }
    match ple { Ple::DelPerfil => del_perfil, Ple::Hq52 => "hq52k", Ple::Hq8 => "hq8k" }
}

/// Importancia por columna de `<dir>/<canónico>.f32`, la convención del v9.1: solo la tienen
/// los `mlp.down`, un f32 por columna de entrada. Vacía para todo lo demás.
fn importancia(dir: Option<&Path>, canonical: &str, cols: usize) -> Result<Vec<f32>> {
    let Some(dir) = dir else { return Ok(Vec::new()) };
    if !canonical.ends_with(".mlp.down.weight") { return Ok(Vec::new()); }
    let path = dir.join(format!("{canonical}.f32"));
    let bytes = std::fs::read(&path).with_context(|| format!("falta la importancia {}", path.display()))?;
    ensure!(bytes.len() == cols * 4, "{canonical}: la importancia trae {} bytes y se esperan {}", bytes.len(), cols * 4);
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn stored_bytes(dtype: &str, elements: u64) -> u64 {
    match dtype {
        "hq8k" => crate::hqs::hq8k_size(elements as usize) as u64,
        "hq52k" => crate::hqs::hq52k_size(elements as usize) as u64,
        "hq44k_g16" => elements / 256 * 152,
        _ => elements * 2,
    }
}

fn emit(w: &mut dyn Write, t: &Planned, raw: &[u8], dtype: &str, imp: &[f32]) -> Result<()> {
    let elements: usize = t.shape.iter().product();
    let width = if dtype == "F32" { 4 } else { 2 };
    ensure!(raw.len() == elements * width, "{}: bytes de origen que no cuadran", t.canonical);
    let row = *t.shape.last().context("forma vacía")?;
    let batch = row * ROWS_PER_BATCH;
    for start in (0..elements).step_by(batch) {
        let end = elements.min(start + batch);
        let values: Vec<f32> = (start..end).map(|i| source_f32(raw, dtype, i)).collect();
        ensure!(values.iter().all(|v| v.is_finite()), "{}: valores no finitos", t.canonical);
        if t.dtype == "hq8k" {
            w.write_all(&crate::hqs::quantize_hq8k(&values))?;
        } else if t.dtype == "hq52k" {
            w.write_all(&crate::hqs::quantize_hq52k(&values))?;
        } else if t.dtype == crate::hqs::v4::DTYPE {
            // `imp` vacía = importancia 1 en todas las columnas, que es como queda sin calibrar.
            w.write_all(&crate::hqs::v4::quantize(&values, row, imp)?)?;
        } else {
            let mut bytes = Vec::with_capacity(values.len() * 2);
            for v in values {
                let y = f16::from_f32(v);
                ensure!(y.is_finite(), "{}: valor fuera de FP16", t.canonical);
                bytes.extend_from_slice(&y.to_le_bytes());
            }
            w.write_all(&bytes)?;
        }
    }
    Ok(())
}

struct Hashing<'a> { inner: &'a mut dyn Write, state: xxhash_rust::xxh3::Xxh3, written: u64 }
impl Write for Hashing<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?; self.state.update(&buf[..n]); self.written += n as u64; Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> { self.inner.flush() }
}

pub fn convert(o: &Options) -> Result<[u8; 16]> {
    ensure!(!o.out.exists(), "{} ya existe", o.out.display());
    let root: serde_json::Value = serde_json::from_slice(&std::fs::read(o.model.join("config.json"))?)?;
    ensure!(root["model_type"] == "gemma4", "requiere un checkpoint Gemma 4");
    ensure!(root["text_config"]["enable_moe_block"] != true, "es un MoE: usa la conversión del MoE");
    let mapper = Gemma4Mapper::from_model_path(o.model, &root)?;
    let reader = SafetensorReader::from_folder(o.model)?;

    // Plan del texto: lo que el mapper reconoce, en el orden del checkpoint por capas.
    let mut plan = Vec::new();
    for (name, info) in reader.iter_tensors() {
        let Some(m) = mapper.map_tensor(name) else { continue };
        ensure!(["BF16", "F16", "F32"].contains(&info.dtype.as_str()), "{name}: dtype {} no admitido", info.dtype);
        let canonical = format!("text.{}", m.canonical_name);
        let dtype = storage(o.profile, o.ple, &canonical, m.quant_hint, &info.shape);
        let elements: u64 = info.shape.iter().map(|&d| d as u64).product();
        let bytes = stored_bytes(dtype, elements);
        plan.push(Planned { canonical, source: name.to_string(), layer: m.layer_idx.map_or(-1, |l| l as i32),
            shape: info.shape.clone(), dtype, bytes });
    }
    plan.sort_by(|a, b| (a.layer, &a.canonical).cmp(&(b.layer, &b.canonical)));
    if let Some(expected) = mapper.expected_tensor_count() {
        ensure!(plan.len() == expected, "{} tensores de texto, el mapper espera {expected}", plan.len());
    }

    // Configuración: los hints de texto del mapper, con el vocabulario del embedding real.
    let mut text = mapper.execution_hints();
    let embedding = plan.iter().find(|t| t.canonical == "text.token_embedding.weight").context("falta el embedding")?;
    text["vocab_size"] = serde_json::json!(embedding.shape[0]);
    let softcap = root["text_config"]["final_logit_softcapping"].as_f64();
    let tower = if o.vision { Some(Tower::plan(o.model)?) } else { None };
    let mut config = build_execution_hints_binary(&text)?;
    if let Some(t) = &tower { config = append_vision(&config, &t.hints)?; }
    let tokenizer = crate::htf::build_htf(o.model)?;
    ensure!(tokenizer.len() > 64, "falta el tokenizer");
    let template = std::fs::read(o.model.join("chat_template.jinja")).ok();
    let base_id = crate::base_id::compute(o.model)?;

    let mut layout = Vec::with_capacity(plan.len());
    let mut end = 0u64;
    for t in &plan { let at = align_up(end, TENSOR_ALIGN); layout.push(at); end = at + t.bytes; }
    let text_size = end;

    let mut hashes = Vec::with_capacity(plan.len());
    let mut builder = Builder::create(o.out)?;
    let (text_at, _) = builder.payload(text_size, 4096, |w| {
        let mut written = 0u64;
        for (t, &at) in plan.iter().zip(&layout) {
            w.write_all(&vec![0u8; (at - written) as usize])?;
            let info = reader.tensor_info(&t.source).unwrap();
            let raw = reader.read_raw(&t.source)?;
            let mut hashing = Hashing { inner: &mut *w, state: xxhash_rust::xxh3::Xxh3::new(), written: 0 };
            let imp = importancia(o.importancia, &t.canonical, *t.shape.last().context("forma vacía")?)?;
            emit(&mut hashing, t, raw, &info.dtype, &imp)?;
            ensure!(hashing.written == t.bytes, "{}: {} bytes escritos, {} previstos", t.canonical, hashing.written, t.bytes);
            hashes.push((hashing.state.digest128().to_le_bytes(), xxhash_rust::xxh3::xxh3_64(raw)));
            written = at + t.bytes;
            eprintln!("{}/{} {} {}", hashes.len(), plan.len(), t.canonical, t.dtype);
        }
        Ok(())
    })?;
    let (tok_at, tok_hash) = builder.payload(tokenizer.len() as u64, 4096, |w| { w.write_all(&tokenizer)?; Ok(()) })?;
    let (cfg_at, cfg_hash) = builder.payload(config.len() as u64, 4096, |w| { w.write_all(&config)?; Ok(()) })?;
    let tpl = match &template {
        Some(bytes) => Some(builder.payload(bytes.len() as u64, 4096, |w| { w.write_all(bytes)?; Ok(()) })?),
        None => None,
    };
    let written = match &tower { Some(t) => Some(t.write(&mut builder)?), None => None };

    let mut md = Metadata::default();
    let mut weight_bytes = 0;
    for (i, t) in plan.iter().enumerate() {
        md.tensors.push(Tensor { name: t.canonical.clone(), component: 1, layer: t.layer, dtype: t.dtype.into(),
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
    if let Some(cap) = softcap {
        md.properties.push((3, Property { key: "final_logit_softcapping".into(), value: Value::Real(cap) }));
    }
    if let (Some(t), Some(w)) = (&tower, &written) { t.register(&mut md, w, 1, 3)?; }
    for i in 0..md.components.len() { md.components[i].composite = md.composite(&md.components[i]); }

    let mut params = vec![
        Property { key: "profile".into(), value: Value::Str(match o.profile { Profile::Q8 => "q8", Profile::Compacto => "compacto" }.into()) },
        Property { key: "ple".into(), value: Value::Str(match o.ple { Ple::DelPerfil => "del perfil", Ple::Hq52 => "hq52k", Ple::Hq8 => "hq8k" }.into()) },
        Property { key: "base_id".into(), value: Value::Str(base_id) },
        Property { key: "source_model".into(), value: Value::Str(o.model.display().to_string()) },
        Property { key: "calibration".into(), value: Value::Str(match o.importancia {
            Some(d) => d.file_name().map_or_else(|| d.display().to_string(), |n| n.to_string_lossy().into_owned()),
            None => "none".into() }) },
    ];
    if let Some(w) = &written {
        params.push(Property { key: "vision_fp16_inexact_values".into(), value: Value::Int(w.inexact as i64) });
    }
    let identity = md.identity();
    md.provenance.push(Step { kind: step::CONVERT, state: step::COMPLETE, tool: "hnf10 convert".into(),
        version: env!("CARGO_PKG_VERSION").into(), commit: o.commit.into(), input: [0; 16], output: identity,
        calibration: String::new(), touched: vec![], params });
    builder.finish(md)?;

    let opened = open(o.out)?;
    ensure!(opened.header.identity == identity, "identidad distinta al reabrir");
    let bad = verify_tensors(o.out, &opened.metadata)?;
    ensure!(bad.is_empty(), "tensores que no cuadran: {bad:?}");
    Ok(identity)
}
