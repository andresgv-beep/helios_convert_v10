//! Gemma 4 MoE source inventory. This is not an HNF execution ABI yet.
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use super::types::{QuantHint, TensorCategory, TensorMapping};

#[derive(Debug, Clone)]
pub struct MoeSpec {
    pub layers: usize,
    pub hidden: usize,
    pub experts: usize,
    pub top_k: usize,
    pub intermediate: usize,
}

impl MoeSpec {
    pub fn from_text_config(t: &Value) -> Result<Option<Self>> {
        if !t.get("enable_moe_block").and_then(Value::as_bool).unwrap_or(false) {
            return Ok(None);
        }
        let positive = |key: &str| -> Result<usize> {
            let n = t.get(key).and_then(Value::as_u64)
                .with_context(|| format!("Gemma 4 MoE missing integer {key}"))?;
            ensure!(n > 0 && n <= u32::MAX as u64, "invalid Gemma 4 MoE {key}: {n}");
            Ok(n as usize)
        };
        let spec = Self {
            layers: positive("num_hidden_layers")?, hidden: positive("hidden_size")?,
            experts: positive("num_experts")?, top_k: positive("top_k_experts")?,
            intermediate: positive("moe_intermediate_size")?,
        };
        ensure!(spec.top_k <= spec.experts, "top_k_experts exceeds num_experts");
        Ok(Some(spec))
    }

    /// Logical source dimensions; no row padding and no merging of experts.
    pub fn required_tensor_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let mut out = Vec::new();
        for layer in 0..self.layers {
            for (suffix, shape) in [
                ("experts.gate_up_proj", vec![self.experts, 2*self.intermediate, self.hidden]),
                ("experts.down_proj", vec![self.experts, self.hidden, self.intermediate]),
                ("router.proj.weight", vec![self.experts, self.hidden]),
                ("router.scale", vec![self.hidden]),
                ("router.per_expert_scale", vec![self.experts]),
                ("pre_feedforward_layernorm_2.weight", vec![self.hidden]),
                ("post_feedforward_layernorm_1.weight", vec![self.hidden]),
                ("post_feedforward_layernorm_2.weight", vec![self.hidden]),
            ] {
                out.push((format!("model.language_model.layers.{layer}.{suffix}"), shape));
            }
        }
        out
    }
}

pub fn map_extra_tensor(layer: usize, suffix: &str) -> Option<TensorMapping> {
    // Packed rank-three tensors retain expert-major order and gate/up halves.
    // Keep router and scales outside low-bit quantization for the initial port.
    let (name, quant, category) = match suffix {
        "experts.gate_up_proj" => ("moe.experts.gate_up.weight", QuantHint::HQ4K, TensorCategory::MoEExpert),
        "experts.down_proj" => ("moe.experts.down.weight", QuantHint::HQ4K, TensorCategory::MoEExpert),
        "router.proj.weight" => ("moe.router.weight", QuantHint::FP16, TensorCategory::MoERouter),
        "router.scale" => ("moe.router.input_scale", QuantHint::FP16, TensorCategory::MoERouter),
        "router.per_expert_scale" => ("moe.router.expert_scale", QuantHint::FP16, TensorCategory::MoERouter),
        "pre_feedforward_layernorm_2.weight" => ("ln_moe_in.weight", QuantHint::FP16, TensorCategory::Norm),
        "post_feedforward_layernorm_1.weight" => ("ln_mlp_branch_post.weight", QuantHint::FP16, TensorCategory::Norm),
        "post_feedforward_layernorm_2.weight" => ("ln_moe_branch_post.weight", QuantHint::FP16, TensorCategory::Norm),
        _ => return None,
    };
    Some(TensorMapping::new(format!("layer{layer}.{name}"), quant, category).with_layer(layer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn config() -> Value { json!({"enable_moe_block":true,"num_hidden_layers":30,
        "hidden_size":2816,"num_experts":128,"top_k_experts":8,"moe_intermediate_size":704}) }
    #[test]
    fn source_shapes_preserve_expert_axis_and_gate_up_halves() {
        let s = MoeSpec::from_text_config(&config()).unwrap().unwrap();
        let shapes = s.required_tensor_shapes();
        assert_eq!(shapes.len(), 240);
        assert_eq!(shapes[0].1, [128,1408,2816]);
        assert_eq!(shapes[1].1, [128,2816,704]);
        assert_eq!(shapes[3].1, [2816]);
        assert_eq!(shapes[4].1, [128]);
    }
    #[test]
    fn malformed_topology_is_rejected() {
        for (key, value) in [("top_k_experts",json!(129)),("num_experts",json!(0)),
            ("moe_intermediate_size",Value::Null),("hidden_size",json!(-1))] {
            let mut c=config(); c[key]=value;
            assert!(MoeSpec::from_text_config(&c).is_err());
        }
        assert!(MoeSpec::from_text_config(&json!({})).unwrap().is_none());
    }
    #[test]
    fn every_extra_tensor_maps_once_with_protected_router() {
        let spec=MoeSpec::from_text_config(&config()).unwrap().unwrap();
        let mut names=std::collections::HashSet::new();
        for (name, _) in spec.required_tensor_shapes() {
            let rest=name.strip_prefix("model.language_model.layers.").unwrap();
            let (layer,suffix)=rest.split_once('.').unwrap();
            let m=map_extra_tensor(layer.parse().unwrap(),suffix).unwrap();
            assert!(names.insert(m.canonical_name));
            if suffix.starts_with("router.") { assert_eq!(m.quant_hint,QuantHint::FP16); }
        }
        assert!(map_extra_tensor(0,"experts.unknown").is_none());
    }
}
