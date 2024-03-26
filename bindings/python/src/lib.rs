use pyo3::create_exception;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use pyo3::types::*;
use tokengeex::ScoredToken;
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
        self.tokenizer
            .encode_batch(texts.iter())
            .map_err(|e| e.into())
    }

    fn encode_ordinary_batch(
        &self,
        texts: Vec<&str>,
    ) -> Result<Vec<Vec<TokenID>>, PyTokenGeeXError> {
        self.tokenizer
            .encode_ordinary_batch(texts.iter())
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
            .decode_batch(ids.iter(), include_special_tokens)
            .map_err(|e| e.into())
    }

    fn token_to_id(&self, token: Token) -> Option<TokenID> {
        self.tokenizer.token_to_id(token)
    }

    fn special_token_to_id(&self, token: &str) -> Option<TokenID> {
        self.tokenizer.special_token_to_id(token)
    }

    fn id_to_token(&self, id: TokenID) -> Option<ScoredToken> {
        self.tokenizer.id_to_token(id)
    }

    fn id_to_special_token(&self, id: TokenID) -> Option<String> {
        self.tokenizer.id_to_special_token(id)
    }

    fn is_special(&self, id: u32) -> Option<bool> {
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

#[pyfunction]
#[pyo3(name = "load")]
fn tokengeex_load_py(filename: &str) -> Result<PyTokenizer, PyTokenGeeXError> {
    let tokenizer =
        tokengeex::load(filename).map_err(std::convert::Into::<PyTokenGeeXError>::into)?;
    Ok(PyTokenizer::from(tokenizer))
}

#[pymodule]
#[pyo3(name = "tokengeex")]
fn tokengeex_module(_: Python, m: &PyModule) -> PyResult<()> {
    // Module: TokenGeeX
    m.add_function(wrap_pyfunction!(tokengeex_load_py, m)?)?;
    m.add_class::<PyTokenizer>()?;

    Ok(())
}
