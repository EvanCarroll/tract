#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Tract(#[from] tract_onnx::prelude::TractError),
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Protobuf(#[from] prost::DecodeError),
    #[error("config.json lacks a numeric `{0}`")]
    Config(&'static str),
    #[error("model input `{0}` is not one the harness knows how to feed")]
    UnexpectedInput(String),
    #[error("model part index {0} is not 0 (encoder), 1 (decoder) or 2 (decoder with past)")]
    UnknownPart(u8),
    #[error("{0} was never loaded")]
    MissingPart(&'static str),
    #[error("rewrite: {0}")]
    Rewrite(String),
}
