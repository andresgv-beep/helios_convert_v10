//! Conversión directa de safetensors a HNFv10 para Gemma 4 MoE.
//!
//! El plan, los codificadores (HQ44, HQ8K, HQ52), la importancia por experto, el HTF3 y la
//! configuración binaria son los del conversor v9.1, usado como biblioteca: cada formato sigue
//! teniendo una sola implementación. Aquí solo cambia el contenedor. Cada tensor va alineado a
//! 4 KiB dentro del componente de texto y su xxh3-128 se calcula mientras se escribe.
use crate::file::{open, verify_tensors, Builder};
use crate::format::{align_up, xxh3_128};
use crate::metadata::{kind, step, Component, Metadata, Property, Step, Tensor, Value};
use crate::migrate::text_param;
use anyhow::{ensure, Context, Result};
use crate::moe_experimental::{emit_tensor_with_importance, make_plan_with, DenseFormat};
use crate::SafetensorReader;
use std::io::Write;
use std::path::Path;

pub struct Options<'a> {
    pub model: &'a Path,
    pub out: &'a Path,
    pub dense: DenseFormat,
    pub importance: Option<&'a Path>,
    pub commit: &'a str,
    pub batch_rows: usize,
}

const TENSOR_ALIGN: u64 = 4096;

struct Hashing<'a> { inner: &'a mut dyn Write, state: xxhash_rust::xxh3::Xxh3, written: u64 }
impl Write for Hashing<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?; self.state.update(&buf[..n]); self.written += n as u64; Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> { self.inner.flush() }
}

fn layer_of(name: &str) -> i32 {
    name.strip_prefix("text.layer").and_then(|r| r.split('.').next()).and_then(|n| n.parse().ok()).unwrap_or(-1)
}

pub fn convert(o: &Options) -> Result<[u8; 16]> {
    ensure!(!o.out.exists(), "{} ya existe", o.out.display());
    let plan = make_plan_with(o.model, o.dense)?;
    let tokenizer = crate::htf::build_htf(o.model)?;
    ensure!(tokenizer.len() > 64, "falta el tokenizer");
    let config = plan.binary_hints()?;
    // El binario tiene que decir lo mismo que la configuración del plan; lo que el binario no
    // trae va como propiedad tipada (hoy, final_logit_softcapping).
    let text_offset = u32::from_le_bytes(config[8..12].try_into().unwrap()) as usize;
    let expected = crate::hints::TextModelConfigBin::from_json(&plan.text).to_bytes();
    ensure!(config.get(text_offset..text_offset + 128) == Some(&expected[..]), "la configuración binaria no coincide con el plan");
    let softcap = plan.text["final_logit_softcapping"].as_f64().context("falta final_logit_softcapping en el plan")?;
    let template = std::fs::read(o.model.join("chat_template.jinja")).ok();
    let base_id = crate::base_id::compute(o.model)?;
    let reader = SafetensorReader::from_folder(o.model)?;

    // Disposición del texto: un tensor tras otro, cada uno alineado a 4 KiB.
    let mut layout = Vec::new();
    let mut end = 0u64;
    for t in &plan.tensors {
        let at = align_up(end, TENSOR_ALIGN);
        layout.push(at);
        end = at + t.bytes;
    }
    let text_size = end;

    let mut importance_files = Vec::new();
    let mut hashes = Vec::with_capacity(plan.tensors.len());
    let mut source_hashes = Vec::with_capacity(plan.tensors.len());
    let mut builder = Builder::create(o.out)?;
    let (text_at, _) = builder.payload(text_size, 4096, |w| {
        let mut written = 0u64;
        for (t, &at) in plan.tensors.iter().zip(&layout) {
            w.write_all(&vec![0u8; (at - written) as usize])?;
            let raw = reader.read_raw(&t.source)?;
            source_hashes.push(xxhash_rust::xxh3::xxh3_64(raw));
            let importance = match (o.importance, t.canonical.find(".moe.experts.")) {
                (Some(dir), Some(pos)) => {
                    let layer = t.canonical.strip_prefix("text.").and_then(|n| n.split('.').next()).context("nombre de experto")?;
                    let kind = t.canonical[pos + ".moe.experts.".len()..].trim_end_matches(".weight");
                    let bytes = std::fs::read(dir.join(format!("{layer}.{kind}.f32"))).with_context(|| format!("importancia de {}", t.canonical))?;
                    let values: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                    ensure!(values.len() == t.physical_shape[0] * t.physical_shape[2], "{}: importancia de {} valores", t.canonical, values.len());
                    ensure!(values.iter().all(|v| v.is_finite() && *v >= 0.0), "{}: importancia no válida", t.canonical);
                    importance_files.push(xxhash_rust::xxh3::xxh3_64(&bytes));
                    Some(values)
                }
                _ => None,
            };
            let mut hashing = Hashing { inner: &mut *w, state: xxhash_rust::xxh3::Xxh3::new(), written: 0 };
            emit_tensor_with_importance(&mut hashing, t, raw, o.batch_rows, importance.as_deref())?;
            ensure!(hashing.written == t.bytes, "{}: {} bytes escritos, {} previstos", t.canonical, hashing.written, t.bytes);
            hashes.push(hashing.state.digest128().to_le_bytes());
            written = at + t.bytes;
            eprintln!("{}/{} {}", hashes.len(), plan.tensors.len(), t.canonical);
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
    for (i, t) in plan.tensors.iter().enumerate() {
        let shape: Vec<u64> = t.physical_shape.iter().map(|&d| d as u64).collect();
        let experts = if t.canonical.contains(".moe.experts.") { u32::try_from(shape[0])? } else { 0 };
        md.tensors.push(Tensor {
            name: t.canonical.clone(), component: 1, layer: layer_of(&t.canonical), dtype: t.dtype.clone(),
            logical_last: *t.logical_shape.last().context("forma vacía")? as u64, shape,
            offset: text_at + layout[i], size: t.bytes, experts, hash: hashes[i],
            source: t.source.clone(), source_hash: source_hashes[i] });
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
    md.properties.push((3, Property { key: "final_logit_softcapping".into(), value: Value::Real(softcap) }));
    for i in 0..md.components.len() { md.components[i].composite = md.composite(&md.components[i]); }

    let mut params = vec![
        Property { key: "dense".into(), value: Value::Str(format!("{:?}", o.dense)) },
        Property { key: "base_id".into(), value: Value::Str(base_id) },
        Property { key: "source_model".into(), value: Value::Str(o.model.display().to_string()) },
        Property { key: "calibration".into(), value: Value::Str(if o.importance.is_some() { "imatrix_expert_importance" } else { "none_unit_importance" }.into()) },
    ];
    if let Some(dir) = o.importance {
        params.push(Property { key: "importance_dir".into(), value: Value::Str(dir.display().to_string()) });
        let all: Vec<u8> = importance_files.iter().flat_map(|h| h.to_le_bytes()).collect();
        params.push(Property { key: "importance_files_xxh3_128".into(), value: Value::Bytes(xxh3_128(&all).to_vec()) });
    }
    if let Some(p) = text_param("experimental", &serde_json::json!(true)) { params.push(p); }
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
