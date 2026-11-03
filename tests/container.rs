//! Contenedor HNFv10: ida y vuelta, cortes simulados, estado «en curso» y validación.
use helios_convert_v10::file::{open, open_for_repair, verify_tensors, Builder, Session};
use helios_convert_v10::format::{Header, State, HEADER_SIZE};
use helios_convert_v10::metadata::{kind, step, Component, Metadata, MemoryFormula, Property, Step, Tensor, Value};
use std::fs::OpenOptions;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

fn temp(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("hnf10-{}-{name}.hnf", std::process::id()));
    let _ = std::fs::remove_file(&path);
    path
}

fn component(id: u32, kind: u32, name: &str, offset: u64, size: u64, first: u32, count: u32, bytes_hash: [u8; 16]) -> Component {
    Component { id, kind, required: true, name: name.into(), payload_offset: offset, payload_size: size, alignment: 4096,
        weight_bytes: size, memory: None, first_tensor: first, tensor_count: count, depends_on: None,
        composite: [0; 16], bytes_hash }
}

/// Un texto con dos tensores (uno MoE de 4 expertos) y una configuración sin tensores.
fn sample(path: &PathBuf) -> Metadata {
    let mut b = Builder::create(path).unwrap();
    let a: Vec<u8> = (0..1000u32).map(|i| i as u8).collect();
    let e: Vec<u8> = (0..4096u32).map(|i| (i * 7) as u8).collect();
    let (text_offset, _) = b.payload(5096, 4096, |w| { w.write_all(&a)?; w.write_all(&e)?; Ok(()) }).unwrap();
    let config = b"TNIH-config".to_vec();
    let (config_offset, config_hash) = b.payload(config.len() as u64, 4096, |w| { w.write_all(&config)?; Ok(()) }).unwrap();
    let tensor = |name: &str, offset: u64, size: u64, bytes: &[u8], experts: u32, shape: Vec<u64>| Tensor {
        name: name.into(), component: 1, layer: 0, dtype: "fp16".into(), logical_last: *shape.last().unwrap(), shape,
        offset, size, experts, hash: helios_convert_v10::format::xxh3_128(bytes), source: format!("model.{name}"), source_hash: 7 };
    let mut md = Metadata::default();
    md.tensors.push(tensor("text.layer0.attn.q_proj.weight", text_offset, 1000, &a, 0, vec![20, 25]));
    md.tensors.push(tensor("text.layer0.moe.experts.down.weight", text_offset + 1000, 4096, &e, 4, vec![4, 16, 32]));
    md.components.push(component(1, kind::TEXT, "texto", text_offset, 5096, 0, 2, [0; 16]));
    md.components.push(component(2, kind::CONFIG, "configuración", config_offset, config.len() as u64, 2, 0, config_hash));
    md.memory.push(MemoryFormula { fixed: 1 << 20, terms: vec![(1, 1, 4096.0), (3, 2, 32.0)], maxima: vec![(1, 16384), (3, 4096)] });
    md.components[0].memory = Some(0);
    md.properties.push((2, Property { key: "final_logit_softcapping".into(), value: Value::Real(30.0) }));
    md.provenance.push(Step { kind: step::CONVERT, state: step::COMPLETE, tool: "prueba".into(), version: "0".into(), commit: "-".into(),
        input: [1; 16], output: [2; 16], calibration: String::new(), touched: vec![], params: vec![Property { key: "dense".into(), value: Value::Str("hq8".into()) }] });
    b.finish(md.clone()).unwrap();
    open(path).unwrap().metadata
}


#[test]
fn round_trip_and_tensor_hashes() {
    let path = temp("ida");
    let md = sample(&path);
    let opened = open(&path).unwrap();
    assert_eq!(opened.slot, 0);
    assert_eq!(opened.metadata, md);
    assert_eq!(md.properties[0].1.value, Value::Real(30.0));
    assert!(verify_tensors(&path, &md).unwrap().is_empty());
    // Un byte cambiado en los pesos solo lo encuentra la verificación profunda.
    let f = OpenOptions::new().write(true).open(&path).unwrap();
    f.write_all_at(&[0xEE], md.tensors[1].offset + 10).unwrap();
    assert!(open(&path).is_ok());
    assert_eq!(verify_tensors(&path, &md).unwrap(), vec![md.tensors[1].name.clone()]);
}

#[test]
fn commit_switches_slot_and_falls_back_on_damage() {
    let path = temp("corte");
    let md = sample(&path);
    let first = open(&path).unwrap();
    let mut sealed = md.clone();
    sealed.provenance.push(Step { kind: step::SEAL, state: step::COMPLETE, tool: "seal".into(), version: "0".into(), commit: "-".into(),
        input: first.header.identity, output: first.header.identity, calibration: String::new(), touched: vec![], params: vec![] });
    Session::begin(&path).unwrap().commit(sealed.clone(), State::Complete).unwrap();
    let second = open(&path).unwrap();
    assert_eq!((second.slot, second.header.sequence), (1, 2));
    assert_eq!(second.metadata.provenance.len(), 2);
    // La procedencia no cambia la identidad.
    assert_eq!(second.header.identity, first.header.identity);

    // Cabecera B rota: manda otra vez la A.
    let f = OpenOptions::new().write(true).open(&path).unwrap();
    f.write_all_at(&[0xFF; 4], HEADER_SIZE as u64 + 100).unwrap();
    let back = open(&path).unwrap();
    assert_eq!((back.slot, back.metadata.provenance.len()), (0, 1));

    // Metadatos nuevos cortados a medias: la cabecera que los apunta deja de valer.
    let path = temp("cortados");
    let md = sample(&path);
    let first = open(&path).unwrap();
    let _ = first;
    Session::begin(&path).unwrap().commit(md.clone(), State::Complete).unwrap();
    let len = std::fs::metadata(&path).unwrap().len();
    OpenOptions::new().write(true).open(&path).unwrap().set_len(len - 10).unwrap();
    assert_eq!(open(&path).unwrap().header.sequence, 1);
}

#[test]
fn in_progress_rejects_even_with_an_older_complete_header() {
    let path = temp("encurso");
    let md = sample(&path);
    let first = open(&path).unwrap();
    let mut pending = md.clone();
    pending.provenance.push(Step { kind: step::REFINE, state: step::IN_PROGRESS, tool: "hnf_refine".into(), version: "0".into(),
        commit: "-".into(), input: first.header.identity, output: [0; 16], calibration: "fnv".into(),
        touched: vec![md.tensors[1].name.clone()], params: vec![] });
    Session::begin(&path).unwrap().commit(pending, State::InProgress).unwrap();
    let error = open(&path).err().unwrap().to_string();
    assert!(error.contains("en curso"), "{error}");
    let (newest, previous) = open_for_repair(&path).unwrap();
    assert_eq!(newest.header.state, State::InProgress);
    assert_eq!(previous.unwrap().header.state, State::Complete);
    // Una sesión normal no puede empezar sobre un «en curso»; solo la de reparación, que
    // parte de la cabecera vigente y es la única que puede cerrarlo.
    assert!(Session::begin(&path).err().unwrap().to_string().contains("en curso"));
    let mut repair = Session::begin_repair(&path).unwrap();
    assert_eq!(repair.current().header.sequence, 2);
    repair.commit(md.clone(), State::Complete).unwrap();
    assert_eq!(open(&path).unwrap().header.sequence, 3);
}

/// Hallazgo de GPT: una confirmación hecha con un estado antiguo tapaba un «en curso».
#[test]
fn stale_or_concurrent_commits_are_rejected() {
    let path = temp("antigua");
    let md = sample(&path);
    // Dos sesiones a la vez: la segunda no consigue el cerrojo.
    let mut writer = Session::begin(&path).unwrap();
    assert!(Session::begin(&path).err().unwrap().to_string().contains("otro proceso"));
    // Alguien que se salta el cerrojo deja una cabecera más nueva: la sesión lo detecta.
    let current = writer.current().header.clone();
    let newer = Header { sequence: current.sequence + 5, ..current };
    OpenOptions::new().write(true).open(&path).unwrap().write_all_at(&newer.to_bytes(), HEADER_SIZE as u64).unwrap();
    let error = writer.commit(md.clone(), State::Complete).err().unwrap().to_string();
    assert!(error.contains("ya no es la de esta sesión"), "{error}");
    drop(writer);

    // El caso reproducido: «en curso», pesos a medias y luego cerrar con el estado viejo.
    let path = temp("encurso-antigua");
    let md = sample(&path);
    let mut refine = Session::begin(&path).unwrap();
    refine.commit(md.clone(), State::InProgress).unwrap();
    refine.file().write_all_at(&[0xAB; 64], md.tensors[1].offset).unwrap();
    drop(refine);                                   // corte antes de cerrar
    assert!(open(&path).is_err());
    assert!(Session::begin(&path).is_err());        // nadie confirma encima con el estado viejo
}

#[test]
fn validation_catches_structural_errors() {
    let path = temp("valida");
    let md = sample(&path);
    let offset = open(&path).unwrap().header.metadata_offset;
    let mut overlap = md.clone();
    overlap.tensors[1].offset -= 1;
    overlap.components[0].composite = overlap.composite(&overlap.components[0]);
    assert!(overlap.validate(offset).unwrap_err().to_string().contains("solapan"));
    let mut stale = md.clone();
    stale.tensors[0].hash[0] ^= 1;
    assert!(stale.validate(offset).unwrap_err().to_string().contains("compuesto"));
    let mut unknown = md.clone();
    unknown.components[1].kind = 99;
    assert!(unknown.validate(offset).is_err());
    unknown.components[1].required = false;
    assert!(unknown.validate(offset).is_ok());
    // Hallazgo de GPT: un tensor fuera de todo rango de componente y fuera del fichero.
    let mut orphan = md.clone();
    let mut extra = md.tensors[0].clone();
    extra.name = "huérfano".into();
    extra.offset = 1 << 40;
    orphan.tensors.push(extra);
    assert!(orphan.validate(offset).unwrap_err().to_string().contains("rangos"));
    let mut twice = md.clone();
    twice.components[1].first_tensor = 0;
    twice.components[1].tensor_count = 1;
    assert!(twice.validate(offset).is_err());
    // Hallazgo de GPT: fórmulas de memoria sin validar.
    for bad in [vec![(1, 1, f64::NAN)], vec![(1, 1, -1.0)], vec![(9, 1, 1.0)], vec![(2, 1, 1.0)], vec![(1, 7, 1.0)]] {
        let mut memory = md.clone();
        memory.memory[0].terms = bad.clone();
        assert!(memory.validate(offset).is_err(), "{bad:?}");
    }
    let mut zero = md.clone();
    zero.memory[0].maxima.push((1, 5));
    assert!(zero.validate(offset).unwrap_err().to_string().contains("repetido"));
    let mut experts = md.clone();
    experts.tensors[1].experts = 3;
    assert!(experts.validate(offset).unwrap_err().to_string().contains("expertos"));
}
