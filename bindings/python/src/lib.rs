use std::str::FromStr;

use pyo3::create_exception;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use pyo3::types::*;
use tokengeex::{Token, TokenID};

create_exception!(tokengeex, TokenGeeXError, pyo3::exceptions::PyException);

#[pyclass(dict, module = "tokengeex", name = "Tokenizer")]
#[derive(Clone)]
struct PyTokenizer {
    tokenizer: tokengeex::Tokenizer,
}

impl PyTokenizer {
    fn from(tokenizer: tokengeex::Tokenizer) -> Self {
        PyTokenizer { tokenizer }
    }
}

struct PyTokenGeeXError {
    inner: tokengeex::Error,
}

impl From<tokengeex::Error> for PyTokenGeeXError {
    fn from(e: tokengeex::Error) -> Self {
        PyTokenGeeXError { inner: e }
    }
}

impl From<PyTokenGeeXError> for PyErr {
    fn from(e: PyTokenGeeXError) -> PyErr {
        TokenGeeXError::new_err(e.inner.to_string())
    }
}

#[pymethods]
impl PyTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<TokenID>, PyTokenGeeXError> {
        self.tokenizer.encode(text).map_err(|e| e.into())
    }

    fn encode_ordinary(&self, text: &str) -> Result<Vec<TokenID>, PyTokenGeeXError> {
        self.tokenizer.encode_ordinary(text).map_err(|e| e.into())
    }

    fn encode_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<TokenID>>, PyTokenGeeXError> {
        self.tokenizer.encode_batch(texts).map_err(|e| e.into())
    }

    fn encode_ordinary_batch(
        &self,
        texts: Vec<&str>,
    ) -> Result<Vec<Vec<TokenID>>, PyTokenGeeXError> {
        self.tokenizer
            .encode_ordinary_batch(texts)
            .map_err(|e| e.into())
    }

    fn decode(
        &self,
        ids: Vec<TokenID>,
        include_special_tokens: bool,
    ) -> Result<String, PyTokenGeeXError> {
        self.tokenizer
            .decode(&ids, include_special_tokens)
            .map_err(|e| e.into())
    }

    fn decode_batch(
        &self,
        ids: Vec<Vec<TokenID>>,
        include_special_tokens: bool,
    ) -> Result<Vec<String>, PyTokenGeeXError> {
        self.tokenizer
            .decode_batch(ids, include_special_tokens)
            .map_err(|e| e.into())
    }

    fn token_to_id(&self, token: Token) -> Option<TokenID> {
        self.tokenizer.token_to_id(token)
    }

    fn special_token_to_id(&self, token: &str) -> Option<TokenID> {
        self.tokenizer.special_token_to_id(token)
    }

    fn id_to_token(&self, py: Python, id: TokenID) -> Option<(Py<PyBytes>, f64)> {
        let token = self.tokenizer.id_to_token(id);

        if let Some(token) = token {
            Some((PyBytes::new(py, &token.value).into(), token.score))
        } else {
            None
        }
    }

    fn id_to_special_token(&self, id: TokenID) -> Option<String> {
        self.tokenizer.id_to_special_token(id)
    }

    fn is_special(&self, id: u32) -> bool {
        self.tokenizer.is_special(id)
    }

    fn add_special_tokens(&mut self, tokens: Vec<String>) {
        self.tokenizer.add_special_tokens(tokens);
    }

    fn special_tokens(&self) -> Vec<String> {
        self.tokenizer.special_tokens()
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    fn base_vocab_size(&self) -> usize {
        self.tokenizer.base_vocab_size()
    }

    fn special_vocab_size(&self) -> usize {
        self.tokenizer.special_vocab_size()
    }

    fn save(&self, filename: &str) {
        match self.tokenizer.save(filename) {
            Ok(_) => {}
            Err(e) => {
                Python::with_gil(|py| {
                    PyIOError::new_err(e.to_string()).restore(py);
                    assert!(PyErr::occurred(py));
                    drop(PyErr::fetch(py));
                });
            }
        }
    }

    fn common_prefix_search(&self, text: &str) -> Vec<TokenID> {
        let mut buffer = Vec::with_capacity(256);

        let values: Vec<TokenID> = self
            .tokenizer
            .model()
            .common_prefix_search(text.as_bytes(), &mut buffer)
            .map(|(id, _)| id)
            .collect();

        values
    }

    #[staticmethod]
    fn from_str(_py: Python, json: &str) -> Result<PyTokenizer, PyTokenGeeXError> {
        tokengeex::Tokenizer::from_str(json)
            .map(PyTokenizer::from)
            .map_err(|e| e.into())
    }

    #[staticmethod]
    fn from_file(_py: Python, filepath: &str) -> Result<PyTokenizer, PyTokenGeeXError> {
        tokengeex::Tokenizer::from_file(filepath)
            .map(PyTokenizer::from)
            .map_err(|e| e.into())
    }

    #[allow(clippy::inherent_to_string)]
    fn to_string(&self) -> String {
        self.tokenizer.to_string()
    }

    /// Used to pickle the Tokenizer. Useful for sharing a tokenizer between
    /// Python processes.
    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(&self.tokenizer).map_err(|e| {
            pyo3::exceptions::PyException::new_err(format!(
                "Error while attempting to pickle Tokenizer: {}",
                e
            ))
        })?;
        Ok(PyBytes::new(py, data.as_bytes()).to_object(py))
    }

    /// Used to unpickle the Tokenizer. Useful for sharing a tokenizer between
    /// Python processes.
    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.tokenizer = serde_json::from_slice(s.as_bytes()).map_err(|e| {
                    pyo3::exceptions::PyException::new_err(format!(
                        "Error while attempting to unpickle Tokenizer: {}",
                        e
                    ))
                })?;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}

#[pymodule]
#[pyo3(name = "tokengeex")]
fn tokengeex_module(_: Python, m: &PyModule) -> PyResult<()> {
    // Module: TokenGeeX
    m.add_class::<PyTokenizer>()?;

    Ok(())
}
