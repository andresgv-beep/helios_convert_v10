//! HNFv10 pequeños y deterministas para las pruebas del lector del motor (Hector-engine,
//! tests/test_hnf10.cpp). Si cambia este fichero, hay que regenerarlos y actualizar la prueba.
use crate::file::{Builder, Session};
use crate::format::{xxh3_128, State};
use crate::metadata::{kind, step, Component, Metadata, MemoryFormula, Property, Step, Tensor, Value};
use anyhow::Result;
use std::path::Path;

fn bytes(n: usize, seed: u32) -> Vec<u8> { (0..n as u32).map(|i| (i.wrapping_mul(2654435761).wrapping_add(seed) >> 13) as u8).collect() }

/// `small.hnf`: texto (dos tensores, uno MoE de 4 expertos), tokenizer, configuración con una
/// propiedad, plantilla opcional y una fórmula de memoria; sellado una vez, así que manda la
/// cabecera B con secuencia 2. `in_progress.hnf`: lo mismo con una tercera cabecera «en curso».
pub fn write(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    for (name, pending) in [("small.hnf", false), ("in_progress.hnf", true)] {
        let path = dir.join(name);
        let _ = std::fs::remove_file(&path);
        let mut b = Builder::create(&path)?;
        let q = bytes(1024, 1);
        let e = bytes(4 * 512, 2);
        let (text, _) = b.payload((q.len() + e.len()) as u64, 4096, |w| { w.write_all(&q)?; w.write_all(&e)?; Ok(()) })?;
        let tok = b"HTF3 de prueba".to_vec();
        let (tok_at, tok_hash) = b.payload(tok.len() as u64, 4096, |w| { w.write_all(&tok)?; Ok(()) })?;
        let cfg = b"TNIH de prueba".to_vec();
        let (cfg_at, cfg_hash) = b.payload(cfg.len() as u64, 4096, |w| { w.write_all(&cfg)?; Ok(()) })?;
        let tpl = "{{ messages }}\n".as_bytes().to_vec();
        let (tpl_at, tpl_hash) = b.payload(tpl.len() as u64, 4096, |w| { w.write_all(&tpl)?; Ok(()) })?;
        let tensor = |name: &str, layer, offset, data: &[u8], shape: Vec<u64>, experts| Tensor {
            name: name.into(), component: 1, layer, dtype: "fp16".into(), logical_last: *shape.last().unwrap(), shape,
            offset, size: data.len() as u64, experts, hash: xxh3_128(data), source: format!("model.{name}"), source_hash: 42 };
        let component = |id, kind, required, name: &str, offset, size, first, count, hash| Component {
            id, kind, required, name: name.into(), payload_offset: offset, payload_size: size, alignment: 4096,
            weight_bytes: size, memory: None, first_tensor: first, tensor_count: count, depends_on: None,
            composite: [0; 16], bytes_hash: hash };
        let mut md = Metadata::default();
        md.tensors.push(tensor("text.layer0.attn.q_proj.weight", 0, text, &q, vec![16, 32], 0));
        md.tensors.push(tensor("text.layer1.moe.experts.down.weight", 1, text + q.len() as u64, &e, vec![4, 16, 16], 4));
        md.components.push(component(1, kind::TEXT, true, "texto", text, (q.len() + e.len()) as u64, 0, 2, [0; 16]));
        md.components.push(component(2, kind::TOKENIZER, true, "tokenizer", tok_at, tok.len() as u64, 2, 0, tok_hash));
        md.components.push(component(3, kind::CONFIG, true, "configuración", cfg_at, cfg.len() as u64, 2, 0, cfg_hash));
        md.components.push(component(4, kind::TEMPLATE, false, "plantilla", tpl_at, tpl.len() as u64, 2, 0, tpl_hash));
        md.components[0].memory = Some(0);
        md.memory.push(MemoryFormula { fixed: 1 << 20, terms: vec![(1, 1, 128.0), (3, 2, 32.0)], maxima: vec![(1, 4096), (3, 1024)] });
        md.properties.push((3, Property { key: "final_logit_softcapping".into(), value: Value::Real(30.0) }));
        let step_of = |kind, state, tool: &str| Step { kind, state, tool: tool.into(), version: "0".into(), commit: String::new(),
            input: [0; 16], output: [0; 16], calibration: String::new(), touched: vec![], params: vec![] };
        md.provenance.push(step_of(step::CONVERT, step::COMPLETE, "fixtures"));
        b.finish(md.clone())?;
        let mut session = Session::begin(&path)?;
        md.provenance.push(step_of(step::SEAL, step::COMPLETE, "fixtures"));
        session.commit(md.clone(), State::Complete)?;
        if pending {
            let mut refine = step_of(step::REFINE, step::IN_PROGRESS, "fixtures");
            refine.touched = vec![md.tensors[1].name.clone()];
            md.provenance.push(refine);
            session.commit(md, State::InProgress)?;
        }
    }
    Ok(())
}
