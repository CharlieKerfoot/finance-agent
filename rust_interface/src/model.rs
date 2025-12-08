use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor, quantized::gguf_file};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama as model;
use model::ModelWeights;
use tokenizers::Tokenizer;

pub struct TextGeneration {
    model: ModelWeights,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    device: Device,
}

impl TextGeneration {
    pub fn new(model_path: &str) -> Result<Self> {
        let device = Device::Cpu;

        let mut file = std::fs::File::open(model_path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let model = ModelWeights::from_gguf(content, &mut file, &device)?;

        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model("ckerf/arbagent-llama3-8b-lora".to_string());
        let tokenizer_path = repo.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

        let logits_processor = LogitsProcessor::new(299792458, Some(0.7), Some(0.9));

        Ok(Self {
            model,
            tokenizer,
            logits_processor,
            device,
        })
    }

    pub fn run(&mut self, prompt: &str, sample_len: usize) -> Result<String> {
        let tokens = self.tokenizer.encode(prompt, true).map_err(E::msg)?;
        let mut token_ids = tokens.get_ids().to_vec();
        let mut output_text = String::new();

        if token_ids.is_empty() {
            return Err(E::msg("Tokenizer produced 0 tokens"));
        }

        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { token_ids.len() };
            let start_pos = token_ids.len().saturating_sub(context_size);

            let input_ids = &token_ids[start_pos..];
            let input_tensor = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;

            let seq_len_offset = token_ids.len().saturating_sub(input_ids.len());

            let logits = self.model.forward(&input_tensor, seq_len_offset)?;
            let logits = logits.squeeze(0)?;

            let last_token_logits = if logits.rank() == 1 {
                logits
            } else {
                let (seq_len, _) = logits.dims2()?;
                logits.get(seq_len - 1)?
            };

            let next_token = self.logits_processor.sample(&last_token_logits)?;
            token_ids.push(next_token);

            if next_token == 128001 || next_token == 128009 {
                break;
            }

            if let Some(text) = self.tokenizer.id_to_token(next_token) {
                let text = text.replace('Ġ', " ").replace("<0x0A>", "\n");
                output_text.push_str(&text);
            }
        }

        Ok(output_text)
    }
}
