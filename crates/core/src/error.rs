use std::fmt;
use std::io;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    InvalidDimensions { width: u32, height: u32 },
    UnsupportedFormat(String),
    DecodeFailed(String),
    EncodeFailed(String),
    ColorTransform(String),
    SecurityLimit(String),
    InvalidSpec(String),
    Archive(String),
    Io(String),
    MemoryBudgetExceeded { requested: usize, available: usize },
    NotFound(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions { width, height } => {
                write!(f, "invalid image dimensions: {width}x{height}")
            }
            Self::UnsupportedFormat(msg) => write!(f, "unsupported format: {msg}"),
            Self::DecodeFailed(msg) => write!(f, "decode failed: {msg}"),
            Self::EncodeFailed(msg) => write!(f, "encode failed: {msg}"),
            Self::ColorTransform(msg) => write!(f, "color transform failed: {msg}"),
            Self::SecurityLimit(msg) => write!(f, "security limit exceeded: {msg}"),
            Self::InvalidSpec(msg) => write!(f, "invalid target specification: {msg}"),
            Self::Archive(msg) => write!(f, "archive error: {msg}"),
            Self::Io(msg) => write!(f, "io error: {msg}"),
            Self::MemoryBudgetExceeded { requested, available } => {
                write!(
                    f,
                    "memory budget exceeded: requested {requested} bytes with {available} bytes free"
                )
            }
            Self::NotFound(msg) => write!(f, "not found: {msg}"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<io::Error> for EngineError {
    fn from(value: io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

pub type EngineResult<T> = Result<T, EngineError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Warning,
    Error,
    Fatal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub severity: Severity,
    pub source: String,
    pub message: String,
}

impl Diagnostic {
    pub fn warning(source: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Warning,
            source: source.into(),
            message: message.into(),
        }
    }

    pub fn error(source: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            source: source.into(),
            message: message.into(),
        }
    }
}
