//! Metadatos binarios de HNFv10 (docs/HNFV10.md §3–§10): codificar, decodificar y validar.
use crate::format::{xxh3_128, ALIGN, PAYLOAD_START};
use anyhow::{bail, ensure, Context, Result};
use std::collections::{BTreeSet, HashMap};

pub const META_MAGIC: &[u8; 4] = b"HNFM";
pub const META_VERSION: u32 = 1;
const OPTIONAL_SECTION: u32 = 1 << 31;
const NONE: u32 = u32::MAX;

pub mod kind {
    pub const TEXT: u32 = 1;
    pub const VISION: u32 = 2;
    pub const AUDIO: u32 = 3;
    pub const VIDEO: u32 = 4;
    pub const TOKENIZER: u32 = 16;
    pub const CONFIG: u32 = 17;
    pub const TEMPLATE: u32 = 18;
    pub fn known(k: u32) -> bool { matches!(k, TEXT | VISION | AUDIO | VIDEO | TOKENIZER | CONFIG | TEMPLATE) }
}

pub mod step {
    pub const CONVERT: u32 = 1;
    pub const CAPTURE_IMPORTANCE: u32 = 2;
    pub const REFINE: u32 = 3;
    pub const SEAL: u32 = 4;
    pub const MIGRATE: u32 = 5;
    pub const COMPLETE: u32 = 1;
    pub const IN_PROGRESS: u32 = 2;
    pub const CANCELLED: u32 = 3;
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value { Bool(bool), Int(i64), Real(f64), Str(String), Bytes(Vec<u8>) }

#[derive(Clone, Debug, PartialEq)]
pub struct Property { pub key: String, pub value: Value }

#[derive(Clone, Debug, PartialEq)]
pub struct Component {
    pub id: u32,
    pub kind: u32,
    pub required: bool,
    pub name: String,
    pub payload_offset: u64,
    pub payload_size: u64,
    pub alignment: u32,
    pub weight_bytes: u64,
    pub memory: Option<u32>,
    pub first_tensor: u32,
    pub tensor_count: u32,
    pub depends_on: Option<u32>,
    pub composite: [u8; 16],
    pub bytes_hash: [u8; 16],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub component: u32,
    pub layer: i32,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub logical_last: u64,
    pub offset: u64,
    pub size: u64,
    pub experts: u32,
    pub hash: [u8; 16],
    pub source: String,
    pub source_hash: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryFormula {
    pub fixed: u64,
    /// (variable, exponente, bytes por unidad)
    pub terms: Vec<(u32, u32, f64)>,
    /// (variable, máximo admitido)
    pub maxima: Vec<(u32, u64)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub kind: u32,
    pub state: u32,
    pub tool: String,
    pub version: String,
    pub commit: String,
    pub input: [u8; 16],
    pub output: [u8; 16],
    pub calibration: String,
    pub touched: Vec<String>,
    pub params: Vec<Property>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Metadata {
    pub components: Vec<Component>,
    pub tensors: Vec<Tensor>,
    pub memory: Vec<MemoryFormula>,
    pub provenance: Vec<Step>,
    /// (id de componente, propiedad)
    pub properties: Vec<(u32, Property)>,
}

// ---------------------------------------------------------------------------
// Codificación

#[derive(Default)]
struct Strings { bytes: Vec<u8>, seen: HashMap<String, (u32, u32)> }
impl Strings {
    fn add(&mut self, s: &str) -> (u32, u32) {
        if let Some(r) = self.seen.get(s) { return *r; }
        let r = (self.bytes.len() as u32, s.len() as u32);
        self.bytes.extend_from_slice(s.as_bytes());
        self.seen.insert(s.to_string(), r);
        r
    }
}

struct Out { b: Vec<u8> }
impl Out {
    fn u8(&mut self, v: u8) { self.b.push(v); }
    fn u32(&mut self, v: u32) { self.b.extend_from_slice(&v.to_le_bytes()); }
    fn i32(&mut self, v: i32) { self.b.extend_from_slice(&v.to_le_bytes()); }
    fn u64(&mut self, v: u64) { self.b.extend_from_slice(&v.to_le_bytes()); }
    fn i64(&mut self, v: i64) { self.b.extend_from_slice(&v.to_le_bytes()); }
    fn f64(&mut self, v: f64) { self.b.extend_from_slice(&v.to_le_bytes()); }
    fn h(&mut self, v: &[u8; 16]) { self.b.extend_from_slice(v); }
    fn s(&mut self, strings: &mut Strings, v: &str) { let (o, l) = strings.add(v); self.u32(o); self.u32(l); }
}

fn encode_property(out: &mut Out, strings: &mut Strings, p: &Property) {
    out.s(strings, &p.key);
    match &p.value {
        Value::Bool(v) => { out.u8(1); out.u8(*v as u8); }
        Value::Int(v) => { out.u8(2); out.i64(*v); }
        Value::Real(v) => { out.u8(3); out.f64(*v); }
        Value::Str(v) => { out.u8(4); out.s(strings, v); }
        Value::Bytes(v) => { out.u8(5); out.u32(v.len() as u32); out.b.extend_from_slice(v); }
    }
}

impl Metadata {
    pub fn encode(&self) -> Vec<u8> {
        let mut strings = Strings::default();
        let mut sections: Vec<(u32, Vec<u8>)> = Vec::new();

        let mut c = Out { b: Vec::new() };
        c.u32(self.components.len() as u32);
        for x in &self.components {
            c.u32(x.id); c.u32(x.kind); c.u32(x.required as u32); c.s(&mut strings, &x.name);
            c.u64(x.payload_offset); c.u64(x.payload_size); c.u32(x.alignment); c.u32(0);
            c.u64(x.weight_bytes); c.u32(x.memory.unwrap_or(NONE));
            c.u32(x.first_tensor); c.u32(x.tensor_count); c.u32(x.depends_on.unwrap_or(NONE));
            c.h(&x.composite); c.h(&x.bytes_hash);
        }
        sections.push((2, c.b));

        let mut t = Out { b: Vec::new() };
        t.u32(self.tensors.len() as u32);
        for x in &self.tensors {
            t.s(&mut strings, &x.name); t.u32(x.component); t.i32(x.layer); t.s(&mut strings, &x.dtype);
            t.u32(x.shape.len() as u32); t.u32(0);
            for i in 0..4 { t.u64(x.shape.get(i).copied().unwrap_or(0)); }
            t.u64(x.logical_last); t.u64(x.offset); t.u64(x.size); t.u32(x.experts); t.u32(0);
            t.h(&x.hash); t.s(&mut strings, &x.source); t.u64(x.source_hash);
        }
        sections.push((3, t.b));

        let mut m = Out { b: Vec::new() };
        m.u32(self.memory.len() as u32);
        for f in &self.memory {
            m.u64(f.fixed); m.u32(f.terms.len() as u32); m.u32(f.maxima.len() as u32);
            for (v, p, k) in &f.terms { m.u32(*v); m.u32(*p); m.f64(*k); }
            for (v, x) in &f.maxima { m.u32(*v); m.u32(0); m.u64(*x); }
        }
        sections.push((4, m.b));

        let mut p = Out { b: Vec::new() };
        p.u32(self.provenance.len() as u32);
        for s in &self.provenance {
            p.u32(s.kind); p.u32(s.state); p.s(&mut strings, &s.tool); p.s(&mut strings, &s.version); p.s(&mut strings, &s.commit);
            p.h(&s.input); p.h(&s.output); p.s(&mut strings, &s.calibration);
            p.u32(s.touched.len() as u32); for n in &s.touched { p.s(&mut strings, n); }
            p.u32(s.params.len() as u32); for q in &s.params { encode_property(&mut p, &mut strings, q); }
        }
        sections.push((5, p.b));

        let mut q = Out { b: Vec::new() };
        q.u32(self.properties.len() as u32);
        for (component, prop) in &self.properties { q.u32(*component); encode_property(&mut q, &mut strings, prop); }
        sections.push((6, q.b));

        sections.insert(0, (1, strings.bytes));
        let table = 16 + 24 * sections.len();
        let mut out = Out { b: Vec::new() };
        out.b.extend_from_slice(META_MAGIC); out.u32(META_VERSION); out.u32(sections.len() as u32); out.u32(0);
        let mut offset = table as u64;
        for (kind, body) in &sections { out.u32(*kind); out.u32(0); out.u64(offset); out.u64(body.len() as u64); offset += body.len() as u64; }
        for (_, body) in sections { out.b.extend_from_slice(&body); }
        out.b
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.len() >= 16 && &bytes[0..4] == META_MAGIC, "metadatos sin magic HNFM");
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        ensure!(u32_at(4) == META_VERSION, "versión de metadatos {} no soportada", u32_at(4));
        let count = u32_at(8) as usize;
        ensure!(16 + 24 * count <= bytes.len(), "tabla de secciones truncada");
        let mut sections: HashMap<u32, &[u8]> = HashMap::new();
        for i in 0..count {
            let at = 16 + 24 * i;
            let kind = u32_at(at);
            let offset = u64::from_le_bytes(bytes[at + 8..at + 16].try_into().unwrap()) as usize;
            let size = u64::from_le_bytes(bytes[at + 16..at + 24].try_into().unwrap()) as usize;
            let body = bytes.get(offset..offset.checked_add(size).context("sección desbordada")?).context("sección fuera de los metadatos")?;
            match kind {
                1..=6 => { ensure!(sections.insert(kind, body).is_none(), "sección {kind} repetida"); }
                k if k & OPTIONAL_SECTION != 0 => {}
                k => bail!("sección obligatoria desconocida {k}"),
            }
        }
        let strings = *sections.get(&1).context("falta la sección STRINGS")?;
        let mut md = Metadata::default();
        {
            let mut r = In::new(sections.get(&2).context("falta COMPONENTS")?, strings);
            for _ in 0..r.u32()? {
                let id = r.u32()?; let kind = r.u32()?; let required = r.u32()? & 1 != 0; let name = r.s()?;
                let payload_offset = r.u64()?; let payload_size = r.u64()?; let alignment = r.u32()?; r.u32()?;
                let weight_bytes = r.u64()?; let memory = r.opt()?; let first_tensor = r.u32()?; let tensor_count = r.u32()?;
                let depends_on = r.opt()?; let composite = r.h()?; let bytes_hash = r.h()?;
                md.components.push(Component { id, kind, required, name, payload_offset, payload_size, alignment, weight_bytes,
                    memory, first_tensor, tensor_count, depends_on, composite, bytes_hash });
            }
            r.end()?;
        }
        {
            let mut r = In::new(sections.get(&3).context("falta TENSORS")?, strings);
            for _ in 0..r.u32()? {
                let name = r.s()?; let component = r.u32()?; let layer = r.u32()? as i32; let dtype = r.s()?;
                let rank = r.u32()? as usize; r.u32()?;
                ensure!((1..=4).contains(&rank), "{name}: rango {rank} no admitido");
                let mut shape = Vec::new(); for i in 0..4 { let d = r.u64()?; if i < rank { shape.push(d); } }
                let logical_last = r.u64()?; let offset = r.u64()?; let size = r.u64()?; let experts = r.u32()?; r.u32()?;
                let hash = r.h()?; let source = r.s()?; let source_hash = r.u64()?;
                md.tensors.push(Tensor { name, component, layer, dtype, shape, logical_last, offset, size, experts, hash, source, source_hash });
            }
            r.end()?;
        }
        {
            let mut r = In::new(sections.get(&4).context("falta MEMORY")?, strings);
            for _ in 0..r.u32()? {
                let fixed = r.u64()?; let nt = r.u32()?; let nm = r.u32()?;
                let mut terms = Vec::new(); for _ in 0..nt { let v = r.u32()?; let p = r.u32()?; terms.push((v, p, r.f64()?)); }
                let mut maxima = Vec::new(); for _ in 0..nm { let v = r.u32()?; r.u32()?; maxima.push((v, r.u64()?)); }
                md.memory.push(MemoryFormula { fixed, terms, maxima });
            }
            r.end()?;
        }
        {
            let mut r = In::new(sections.get(&5).context("falta PROVENANCE")?, strings);
            for _ in 0..r.u32()? {
                let kind = r.u32()?; let state = r.u32()?; let tool = r.s()?; let version = r.s()?; let commit = r.s()?;
                let input = r.h()?; let output = r.h()?; let calibration = r.s()?;
                let mut touched = Vec::new(); for _ in 0..r.u32()? { touched.push(r.s()?); }
                let mut params = Vec::new(); for _ in 0..r.u32()? { params.push(r.property()?); }
                md.provenance.push(Step { kind, state, tool, version, commit, input, output, calibration, touched, params });
            }
            r.end()?;
        }
        {
            let mut r = In::new(sections.get(&6).context("falta PROPERTIES")?, strings);
            for _ in 0..r.u32()? { let c = r.u32()?; md.properties.push((c, r.property()?)); }
            r.end()?;
        }
        Ok(md)
    }

    /// Hash compuesto de un componente: sus tensores en orden (§6).
    pub fn composite(&self, c: &Component) -> [u8; 16] {
        let mut buf = Vec::new();
        for t in &self.tensors[c.first_tensor as usize..(c.first_tensor + c.tensor_count) as usize] {
            buf.extend_from_slice(&t.offset.to_le_bytes()); buf.extend_from_slice(&t.size.to_le_bytes()); buf.extend_from_slice(&t.hash);
        }
        xxh3_128(&buf)
    }

    /// Identidad del fichero: metadatos sin procedencia y compuestos (§6).
    pub fn identity(&self) -> [u8; 16] {
        let mut without = self.clone();
        without.provenance.clear();
        let mut buf = xxh3_128(&without.encode()).to_vec();
        for c in &self.components { buf.extend_from_slice(&c.composite); buf.extend_from_slice(&c.bytes_hash); }
        xxh3_128(&buf)
    }

    /// Bytes que ocupan `elements` pesos en `dtype`, o None si el dtype no se conoce.
    pub fn bytes_for(dtype: &str, elements: u64) -> Option<u64> {
        let block = |bytes: u64| (elements % 256 == 0).then(|| elements / 256 * bytes);
        match dtype {
            "fp32" | "int32" => Some(elements * 4),
            "fp16" | "bf16" => Some(elements * 2),
            "int8" => Some(elements),
            "hq44k_g16" => block(152),
            "hq52k" => block(184),
            "hq62k" => block(264),
            "hq8k" => block(272),
            _ => None,
        }
    }

    /// Todo lo que el lector comprueba siempre (§2, §4–§6), sin leer pesos.
    pub fn validate(&self, metadata_offset: u64) -> Result<()> {
        let mut ids = BTreeSet::new();
        let mut ranges = Vec::new();
        for c in &self.components {
            ensure!(ids.insert(c.id), "componente {} repetido", c.id);
            ensure!(kind::known(c.kind) || !c.required, "componente obligatorio {} de tipo desconocido {}", c.name, c.kind);
            ensure!(c.alignment.is_power_of_two() && u64::from(c.alignment) >= 32, "{}: alineación {}", c.name, c.alignment);
            ensure!(c.payload_size == 0 || (c.payload_offset >= PAYLOAD_START && c.payload_offset % u64::from(c.alignment) == 0),
                "{}: payload mal alineado", c.name);
            let end = c.payload_offset.checked_add(c.payload_size).context("payload desbordado")?;
            ensure!(end <= metadata_offset, "{}: payload solapa los metadatos", c.name);
            if c.payload_size > 0 { ranges.push((c.payload_offset, end, c.name.clone())); }
            let last = c.first_tensor.checked_add(c.tensor_count).context("tensores desbordados")?;
            ensure!(last as usize <= self.tensors.len(), "{}: tensores fuera del índice", c.name);
            if let Some(m) = c.memory { ensure!((m as usize) < self.memory.len(), "{}: fórmula de memoria inexistente", c.name); }
            ensure!(self.composite(c) == c.composite, "{}: el compuesto no cuadra con sus tensores", c.name);
        }
        for c in &self.components {
            if let Some(d) = c.depends_on { ensure!(ids.contains(&d) && d != c.id, "{}: depende de un componente inexistente", c.name); }
            for (i, t) in self.tensors[c.first_tensor as usize..(c.first_tensor + c.tensor_count) as usize].iter().enumerate() {
                ensure!(t.component == c.id, "{}: tensor {} asignado a otro componente", c.name, c.first_tensor as usize + i);
                ensure!(t.offset >= c.payload_offset && t.offset + t.size <= c.payload_offset + c.payload_size,
                    "{} fuera del payload de {}", t.name, c.name);
                ensure!(t.experts == 0 || (t.shape[0] == u64::from(t.experts) && t.size % u64::from(t.experts) == 0),
                    "{}: expertos incoherentes", t.name);
            }
        }
        ranges.sort();
        for w in ranges.windows(2) { ensure!(w[0].1 <= w[1].0, "{} y {} se solapan", w[0].2, w[1].2); }
        // Cada tensor, en exactamente un rango de componente y dentro de la zona de payloads.
        let mut owners = vec![0u32; self.tensors.len()];
        for c in &self.components {
            for owner in &mut owners[c.first_tensor as usize..(c.first_tensor + c.tensor_count) as usize] { *owner += 1; }
        }
        if let Some(i) = owners.iter().position(|&n| n != 1) {
            bail!("el tensor {} está en {} rangos de componente: debe estar en uno", self.tensors[i].name, owners[i]);
        }
        for t in &self.tensors {
            let end = t.offset.checked_add(t.size).context("tensor desbordado")?;
            ensure!(t.offset >= PAYLOAD_START && end <= metadata_offset, "{} fuera de la zona de payloads", t.name);
        }
        // Fórmulas de memoria: variables conocidas, exponentes 1–3, coeficientes finitos y no
        // negativos, y un máximo para cada variable que se use.
        for (i, f) in self.memory.iter().enumerate() {
            let mut maxima = BTreeSet::new();
            for (v, max) in &f.maxima {
                ensure!((1..=4).contains(v), "fórmula {i}: variable desconocida {v}");
                ensure!(*max > 0, "fórmula {i}: máximo nulo para la variable {v}");
                ensure!(maxima.insert(*v), "fórmula {i}: máximo repetido para la variable {v}");
            }
            for (v, power, coefficient) in &f.terms {
                ensure!((1..=4).contains(v), "fórmula {i}: variable desconocida {v}");
                ensure!((1..=3).contains(power), "fórmula {i}: exponente {power} no admitido");
                ensure!(coefficient.is_finite() && *coefficient >= 0.0, "fórmula {i}: coeficiente {coefficient} no válido");
                ensure!(maxima.contains(v), "fórmula {i}: la variable {v} no declara su máximo");
            }
        }
        let mut tensors: Vec<_> = self.tensors.iter().map(|t| (t.offset, t.offset + t.size, &t.name)).collect();
        tensors.sort();
        for w in tensors.windows(2) { ensure!(w[0].1 <= w[1].0, "{} y {} se solapan", w[0].2, w[1].2); }
        ensure!(self.tensors.iter().all(|t| ids.contains(&t.component)), "tensor de un componente inexistente");
        // Nombres únicos y tamaño coherente con dtype y forma física.
        let mut names = BTreeSet::new();
        for t in &self.tensors {
            ensure!(names.insert(t.name.as_str()), "tensor {} repetido", t.name);
            let elements = t.shape.iter().try_fold(1u64, |a, &d| a.checked_mul(d)).context("forma desbordada")?;
            let expected = Self::bytes_for(&t.dtype, elements).with_context(|| format!("{}: dtype {} desconocido o forma no divisible", t.name, t.dtype))?;
            ensure!(expected == t.size, "{}: {} bytes para {:?} en {}, se esperaban {}", t.name, t.size, t.shape, t.dtype, expected);
            ensure!(t.logical_last <= *t.shape.last().unwrap_or(&0), "{}: más columnas lógicas que físicas", t.name);
        }
        for (c, p) in &self.properties { ensure!(ids.contains(c), "propiedad {} de un componente inexistente", p.key); }
        ensure!(metadata_offset % ALIGN == 0, "metadatos mal alineados");
        Ok(())
    }
}

struct In<'a> { b: &'a [u8], at: usize, strings: &'a [u8] }
impl<'a> In<'a> {
    fn new(b: &'a [u8], strings: &'a [u8]) -> Self { In { b, at: 0, strings } }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let v = self.b.get(self.at..self.at + n).context("sección truncada")?; self.at += n; Ok(v)
    }
    fn u8(&mut self) -> Result<u8> { Ok(self.take(1)?[0]) }
    fn u32(&mut self) -> Result<u32> { Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
    fn u64(&mut self) -> Result<u64> { Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap())) }
    fn f64(&mut self) -> Result<f64> { Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap())) }
    fn h(&mut self) -> Result<[u8; 16]> { Ok(self.take(16)?.try_into().unwrap()) }
    fn opt(&mut self) -> Result<Option<u32>> { let v = self.u32()?; Ok((v != NONE).then_some(v)) }
    fn s(&mut self) -> Result<String> {
        let (o, l) = (self.u32()? as usize, self.u32()? as usize);
        let raw = self.strings.get(o..o + l).context("cadena fuera de STRINGS")?;
        Ok(std::str::from_utf8(raw).context("cadena no UTF-8")?.to_string())
    }
    fn property(&mut self) -> Result<Property> {
        let key = self.s()?;
        let value = match self.u8()? {
            1 => Value::Bool(self.u8()? != 0),
            2 => Value::Int(self.u64()? as i64),
            3 => Value::Real(self.f64()?),
            4 => Value::Str(self.s()?),
            5 => { let n = self.u32()? as usize; Value::Bytes(self.take(n)?.to_vec()) }
            t => bail!("propiedad {key} de tipo desconocido {t}"),
        };
        Ok(Property { key, value })
    }
    fn end(&self) -> Result<()> { ensure!(self.at == self.b.len(), "bytes sobrantes en una sección"); Ok(()) }
}
