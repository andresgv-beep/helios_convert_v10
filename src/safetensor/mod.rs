// src/safetensor/mod.rs
// ============================================================================
// SAFETENSOR READER - Lee modelos HuggingFace
// ============================================================================

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;
use serde::Deserialize;

/// Información de un tensor en el archivo safetensor
#[derive(Debug, Clone, Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

/// Header del archivo safetensor
#[derive(Debug, Deserialize)]
pub struct SafetensorHeader {
    #[serde(flatten)]
    pub tensors: HashMap<String, TensorInfo>,
    #[serde(rename = "__metadata__")]
    pub metadata: Option<HashMap<String, String>>,
}

/// Archivo safetensor abierto
pub struct SafetensorFile {
    pub path: PathBuf,
    pub header: SafetensorHeader,
    pub header_size: usize,
    mmap: Mmap,
}

impl SafetensorFile {
    /// Abre un archivo safetensor
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)
            .with_context(|| format!("Cannot open {}", path.display()))?;
        
        // Leer tamaño del header (primeros 8 bytes, little-endian u64)
        let mut reader = BufReader::new(&file);
        let mut header_size_bytes = [0u8; 8];
        reader.read_exact(&mut header_size_bytes)?;
        let header_size = u64::from_le_bytes(header_size_bytes) as usize;
        
        // Leer header JSON
        let mut header_bytes = vec![0u8; header_size];
        reader.read_exact(&mut header_bytes)?;
        
        let header: SafetensorHeader = serde_json::from_slice(&header_bytes)
            .with_context(|| "Invalid safetensor header JSON")?;
        
        // Memory map el archivo
        let mmap = unsafe { Mmap::map(&file)? };
        
        Ok(Self {
            path,
            header,
            header_size: 8 + header_size,
            mmap,
        })
    }
    
    /// Lista nombres de tensores
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.header.tensors.keys().map(|s| s.as_str())
    }
    
    /// Obtiene información de un tensor
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.header.tensors.get(name)
    }
    
    /// Lee un tensor como bytes raw
    pub fn read_raw(&self, name: &str) -> Result<&[u8]> {
        let info = self.header.tensors.get(name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found", name))?;
        
        let start = self.header_size + info.data_offsets[0];
        let end = self.header_size + info.data_offsets[1];
        
        Ok(&self.mmap[start..end])
    }
    
    /// Lee un tensor como f32 (convierte desde dtype original)
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.tensor_info(name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found", name))?;
        let data = self.read_raw(name)?;
        
        match info.dtype.as_str() {
            "F32" => {
                Ok(data.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect())
            }
            "F16" => {
                Ok(data.chunks_exact(2)
                    .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect())
            }
            "BF16" => {
                Ok(data.chunks_exact(2)
                    .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect())
            }
            dtype => Err(anyhow!("Unsupported dtype: {}", dtype)),
        }
    }
    
    /// Número de elementos de un tensor
    pub fn numel(&self, name: &str) -> Option<usize> {
        self.tensor_info(name)
            .map(|info| info.shape.iter().product())
    }
}

/// Reader para múltiples archivos safetensor (modelos sharded)
pub struct SafetensorReader {
    files: Vec<SafetensorFile>,
    tensor_to_file: HashMap<String, usize>,
}

impl SafetensorReader {
    /// Abre todos los safetensors de un directorio
    pub fn from_folder(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        
        // Buscar archivos .safetensors
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |e| e == "safetensors"))
            .collect();
        
        if paths.is_empty() {
            return Err(anyhow!("No .safetensors files in {}", dir.display()));
        }
        
        // Ordenar para consistencia
        paths.sort();
        
        // Abrir todos
        let mut files = Vec::with_capacity(paths.len());
        let mut tensor_to_file = HashMap::new();
        
        for (idx, path) in paths.iter().enumerate() {
            let file = SafetensorFile::open(path)?;
            
            for name in file.tensor_names() {
                tensor_to_file.insert(name.to_string(), idx);
            }
            
            files.push(file);
        }
        
        Ok(Self { files, tensor_to_file })
    }
    
    /// Número total de tensores
    pub fn len(&self) -> usize {
        self.tensor_to_file.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.tensor_to_file.is_empty()
    }
    
    /// Iterador sobre todos los tensores, en orden alfabetico estable.
    ///
    /// El orden importa: `builder.rs` escribe los tensores al HNF siguiendo este
    /// iterador, asi que fija el layout fisico del fichero. El HashMap de Rust
    /// siembra su hasher al azar en cada proceso, de modo que iterar sus claves
    /// directamente hacia que dos conversiones identicas produjeran ficheros
    /// distintos (mismos pesos, offsets distintos) y por tanto SHA256 distintos.
    pub fn iter_tensors(&self) -> impl Iterator<Item = (&str, &TensorInfo)> {
        let mut names: Vec<&String> = self.tensor_to_file.keys().collect();
        names.sort_unstable();
        names.into_iter().map(move |name| {
            let file_idx = self.tensor_to_file[name];
            let info = self.files[file_idx].tensor_info(name).unwrap();
            (name.as_str(), info)
        })
    }
    
    /// Lee un tensor como f32
    pub fn read(&self, name: &str) -> Result<Vec<f32>> {
        let file_idx = self.tensor_to_file.get(name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found", name))?;
        self.files[*file_idx].read_f32(name)
    }
    
    /// Lee un tensor como bytes raw
    pub fn read_raw(&self, name: &str) -> Result<&[u8]> {
        let file_idx = self.tensor_to_file.get(name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found", name))?;
        self.files[*file_idx].read_raw(name)
    }
    
    /// Obtiene información de un tensor
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        let file_idx = self.tensor_to_file.get(name)?;
        self.files[*file_idx].tensor_info(name)
    }
    
    /// Shape de un tensor
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensor_info(name).map(|info| info.shape.as_slice())
    }
    
    /// Dtype de un tensor
    pub fn dtype(&self, name: &str) -> Option<&str> {
        self.tensor_info(name).map(|info| info.dtype.as_str())
    }
}
